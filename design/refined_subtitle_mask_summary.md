# 精细化字幕 Mask 功能设计总结

## 1. 背景与目标

原项目的字幕擦除流程主要将 OCR 检测框直接绘制成矩形 Mask，再交给 STTN、LAMA 或 ProPainter 重绘。矩形 Mask 实现简单，但会同时遮盖大量没有字幕的背景像素。对于以静态背景、立绘动画、粒子特效为主的游戏剧情视频，这会增加模型需要重建的区域，容易造成背景结构变化、立绘细节损失和视频闪烁。

本次扩展的核心目标是：

1. 保留原有 OCR 作为字幕位置的粗定位。
2. 在 OCR 框内根据已知字幕颜色生成像素级 Mask。
3. 补回 OCR 容易漏掉的句号、省略号、“一”等字符。
4. 处理逐字显示、半透明边缘以及字幕出现和消失阶段。
5. 将逐帧精细化 Mask 正式接入 STTN、LAMA 和 ProPainter。
6. 保留原矩形 Mask 流程，允许随时关闭新功能并回退。

## 2. 总体架构

```text
输入视频
   │
   ├─ OCR 采样检测
   │    └─ 采样帧 OCR 框覆盖完整采样块
   │
   ├─ 逐帧颜色精细化
   │    ├─ 主字幕颜色模式
   │    ├─ 左侧特殊字幕第二颜色模式
   │    ├─ 主模式颜色引导扩框
   │    └─ 边缘生长、去离群点、闭运算、膨胀
   │
   ├─ 时序 Mask 处理
   │    ├─ 借用未来帧 Mask
   │    ├─ Mask 稳定性检测
   │    └─ 已借入 Mask 连续性保护
   │
   ├─ 非空 Mask PNG 压缩缓存
   │    ├─ 可选同步输出 mask_preview.mp4
   │    └─ 按帧解码 Mask
   │
   └─ Inpainting
        ├─ STTN_DET：逐帧 Mask
        ├─ LAMA：逐帧 Mask
        ├─ ProPainter：逐帧 Mask
        └─ 关闭精细化功能时回退原矩形 Mask
```

## 3. 主要新增和修改的模块

| 文件 | 职责 |
| --- | --- |
| `backend/tools/subtitle_mask.py` | 像素级字幕 Mask 的核心生成算法 |
| `backend/tools/subtitle_mask_video.py` | 独立 Mask Preview 调试工具 |
| `backend/tools/refined_mask_runtime.py` | 正式擦除流程的配置读取、时序处理、压缩缓存和同步预览 |
| `subtitle_mask_config.ini` | OCR、颜色 Mask、第二字幕模式、时序和接入开关的集中配置 |
| `backend/tools/subtitle_detect.py` | OCR GPU、阈值覆盖、采样块传播、处理时长等增强 |
| `backend/main.py` | 将精细化 Mask 接入正式擦除流程 |
| `backend/tools/inpaint_tools.py` | 单张旧 Mask 和逐帧 Mask 的统一输入接口 |
| `backend/inpaint/*_inpaint.py` | STTN_DET、LAMA、ProPainter、OpenCV 的逐帧 Mask 支持 |
| `backend/tools/args_handler.py` | 增加 `0~1` 比例 OCR 区域参数 |
| `test/subtitle_mask_checks.py` | Mask、时序、采样、模型输入和 CLI 换算回归检查 |

## 4. OCR 检测增强

### 4.1 GPU 选择

OCR 检测器会根据 `HardwareAccelerator` 的 CUDA 状态选择 `gpu:0` 或 `cpu`，不再固定使用 CPU。

### 4.2 可配置 OCR 参数

以下 PaddleOCR 检测参数可以在 `subtitle_mask_config.ini` 的 `[ocr]` 中覆盖：

- `limit_side_len`：OCR 输入最长边，增大可改善小字符检测，但会增加耗时和显存占用。
- `thresh`：文本像素概率阈值，降低可提高细线和半透明字符召回率。
- `box_thresh`：文本框置信度阈值，降低可改善小标点召回率，但会增加误检。
- `unclip_ratio`：对已经检测到的文本框进行扩张。
- `sample_step`：每隔多少帧执行一次 OCR；`0` 表示按帧率自适应。
- `crop_before_ocr`：按每个字幕区域裁剪后分别执行 OCR。
- `crop_padding`：在裁剪区域四周保留的上下文像素。
- `crop_upscale`：裁剪区域送入 OCR 前的放大倍数。
- `crop_dedup_iou`：重叠区域产生重复框时的 IoU 合并阈值，`0` 表示关闭。

正式擦除开启精细化 Mask 后，会读取这些 OCR 参数。`[ocr] areas/area` 只供独立 Preview 脚本使用；正式擦除的 OCR 区域来自 GUI 选区或 CLI 的 `-r/-c` 参数。

### 4.3 分区域裁剪 OCR

开启 `crop_before_ocr` 后，每个 OCR 区域按以下流程独立处理：

1. 将像素区域向四周增加 `crop_padding`，并裁剪到视频边界内；
2. 按 `crop_upscale` 放大裁剪图，再调用同一个 PaddleOCR 检测器；
3. 使用实际缩放比例将检测框映射回原视频坐标；
4. 仅保留中心点仍在原始未扩边区域内的框；
5. 按 `crop_dedup_iou` 合并重叠区域产生的重复框。

这样可减少区域外 UI 和复杂背景对检测的干扰，同时提高短句、小标点和细字符在 OCR 输入中的相对尺寸。关闭开关时仍执行原来的整帧 OCR 和区域过滤流程，可随时回退。正式擦除使用 GUI/CLI 传入的多个区域，独立 Preview 使用 `[ocr] areas/area`。

### 4.4 采样块传播

为了避免逐帧 OCR 的高耗时，OCR 仍可按间隔采样。一次成功识别得到的框会覆盖其完整采样块。

例如 `sample_step = 3`：

```text
第 1 帧执行 OCR → 框用于第 1、2、3 帧
第 4 帧执行 OCR → 框用于第 4、5、6 帧
```

这里只传播 OCR 框。颜色 Mask 仍然在每一帧上重新计算，因此字幕消失后不会因为传播矩形框而直接产生整块黑色 Mask。

### 4.5 控制台进度输出

OCR 的 `tqdm` 只在终端确实支持动态刷新时显示，避免 IDE 控制台反复打印 `Subtitle Finding`。

## 5. 单帧精细化 Mask 算法

### 5.1 输入与输出

`SubtitleMaskGenerator` 的输入为：

- 当前 BGR 视频帧；
- 当前帧对应的 OCR 框列表，格式为 `(xmin, xmax, ymin, ymax)`；
- `SubtitleMaskConfig` 参数。

输出为与原帧同尺寸的二值 `uint8` Mask：

- `0`：保留区域；
- `255`：需要重绘的字幕区域。

### 5.2 主字幕颜色检测

算法将目标 BGR 字幕颜色和当前 OCR ROI 转换到 OpenCV 的 8 位 Lab 色彩空间，并计算像素与目标颜色的欧氏距离。

像素分为两级：

- 核心像素：`distance <= core_tolerance`；
- 宽松边缘候选：`distance <= edge_tolerance`。

核心像素是高置信字幕像素。宽松候选用于吸收压缩、抗锯齿和半透明边缘。

### 5.3 近似灰度约束

使用三个 BGR 通道中最大值与最小值的差作为灰度接近程度：

```text
channel_spread = max(B, G, R) - min(B, G, R)
```

只有宽松边缘候选需要满足 `channel_spread <= max_channel_spread`。已经满足 `core_tolerance` 的核心像素不受灰度约束，避免亮度较高但带有轻微色偏的字幕核心被错误删除。

`max_channel_spread = -1` 时关闭该约束。

### 5.4 从核心向边缘重建

宽松颜色像素不会直接全部进入 Mask。算法从核心像素开始进行有限次数的形态学生长，只保留能够从核心连通到达的宽松像素。

该设计可以：

- 吸收字幕抗锯齿和半透明边缘；
- 排除 OCR 框内颜色相似但与字幕不连通的背景区域。

生长次数由 `edge_growth_iterations` 控制。

### 5.5 主模式颜色引导扩框

OCR 可能识别出正文但漏掉句号、省略号、“一”等字符。主字幕模式支持从 OCR 框边缘逐步探查：

1. 分别向左、右、上、下取宽度为 `box_expand_step` 的带状区域。
2. 如果带状区域中至少存在 `box_expand_min_core_pixels` 个核心颜色像素，则把框扩展到这些像素的位置。
3. 从新的边界继续探查。
4. 当前探查带没有足够核心像素时停止该方向扩展。
5. 扩展距离受 `box_expand_max_x` 和 `box_expand_max_y` 限制。

这种逐带扩展能够沿相邻标点继续前进，但不会跨过大段没有字幕颜色的空白去吸收远处背景亮点。

`box_padding` 是生成 Mask 时对最终框增加的固定搜索边距；它与迭代式颜色扩框是两个不同概念。

### 5.6 第二字幕颜色模式

针对只出现在画面左侧的深灰色特殊字幕，每个 OCR 框会独立判断是否切换第二模式。

切换条件：

1. `[secondary_mask] enabled = true`；
2. OCR 框中心位于画面左侧 `left_ratio` 范围内；
3. 原始 OCR 框内完全没有主字幕模式的核心像素。

切换后，该框使用独立的：

- `color`；
- `core_tolerance`；
- `edge_tolerance`；
- `max_channel_spread`。

由于深灰色与游戏背景更容易重合，第二模式当前不执行颜色引导扩框，只在原始 OCR 框及公共 `box_padding` 范围内生成 Mask。

如果框内存在任何主模式核心像素，则主模式优先，不会切换到第二模式。

### 5.7 形态学清理和最终膨胀

颜色重建后依次执行：

1. 闭运算：连接小断点并填补细小空洞，由 `close_kernel_size` 控制。
2. 连通域分析：删除面积小于 `min_component_area` 且远离主体文字的离群点。
3. 邻近小连通域保护：距离主要文字不超过 `isolation_distance` 的小标点仍然保留。
4. 最终膨胀：由 `dilation_size` 和 `dilation_iterations` 控制，用于覆盖字幕边缘。

如果所有连通域都很小，算法倾向于保留它们，避免把整句由细小笔画或标点组成的字幕全部删除。

## 6. 时序 Mask 处理

单帧颜色检测无法完整处理逐字显现阶段：最右侧字符可能只有半透明像素，尚未达到核心颜色阈值。因此加入未来帧借用机制。

### 6.1 未来帧合并

当前帧的输出候选为：

```text
当前 raw_mask OR 后续 N 帧 raw_mask
```

其中 `N = future_mask_frames`。这样可以使用字符完全显现后的 Mask 覆盖它刚开始出现时的半透明阶段。

### 6.2 稳定性检测

如果字幕已经稳定显示，持续引入未来帧可能会把下一句字幕提前带入。算法使用相邻非空 Mask 的 IoU 判断稳定性：

```text
IoU = intersection(mask_t, mask_t-1) / union(mask_t, mask_t-1)
```

当 IoU 连续达到 `mask_stability_iou`，并保持 `mask_stability_frames` 帧后，当前稳定字幕停止吸收新的未来 Mask。

`mask_stability_frames = 0` 时自动使用 `future_mask_frames`。

空 Mask 不计为稳定状态。

### 6.3 已借入 Mask 连续性保护

仅在稳定阈值处停止未来合并，可能导致前一帧已经借入的像素在下一帧突然消失。开启 `preserve_future_mask_continuity` 后：

- 已经借入且仍被当前未来窗口支持的像素会继续保留；
- 稳定字幕不会因此吸收下一句中新出现的像素。

该逻辑解决了字幕初显时 Mask “提前出现 → 中途消失 → 完全显示后重新出现”的跳变问题。

## 7. 独立 Mask Preview 工具

运行：

```bash
python -m backend.tools.subtitle_mask_video
```

或指定配置文件：

```bash
python -m backend.tools.subtitle_mask_video --config subtitle_mask_config.ini
```

该工具执行：

1. 使用项目原有 OCR 流程获得框；
2. 逐帧生成精细化和时序 Mask；
3. 将 Mask 区域涂黑；
4. 可选输出纯二值 Mask 视频；
5. 可选只处理视频开头 `duration_seconds` 秒。

预览框颜色：

- 绿色：原始 OCR 框；
- 橙色：主模式颜色引导扩展后的框。

这可以区分“OCR 没有检测到”与“OCR 已检测但颜色 Mask 漏掉”两类问题。

## 8. 正式擦除流程接入

### 8.1 开关与回退

```ini
[integration]
use_refined_mask = true
```

- `true`：STTN_DET、LAMA、ProPainter 和 OpenCV 使用逐帧精细化 Mask。
- `false`：完整回退到原项目的 OCR 矩形 Mask。

STTN_AUTO 的设计是跳过 OCR 并对手工选择区域进行智能擦除，因此不接入 OCR 精细化 Mask。需要测试 STTN 与精细化 Mask 的组合时，应选择 STTN_DET。

### 8.2 稀疏压缩缓存

正式流程在 OCR 后增加一次逐帧 Mask 生成。为了避免把整段视频的全分辨率二维数组常驻内存：

- 只缓存非空 Mask；
- 每帧 Mask 使用 PNG 无损压缩；
- 帧号使用与 OCR 字典一致的 1-based 编号；
- 模型批处理时按帧号解码；
- 不在缓存中的帧返回同尺寸全零 Mask。

未来帧缓冲区只保留 `future_mask_frames` 附近的原始 Mask；只有开启同步预览时才同时保留对应原视频帧。

### 8.3 模型的逐帧 Mask 兼容

`normalize_frame_masks()` 统一接受两种输入：

1. 原流程的一张二维 Mask：自动复制给批次中的所有帧；
2. 与帧数量相同的 Mask 列表：每帧保留独立 Mask。

因此关闭新功能时，模型接口仍兼容原有调用方式。

模型裁剪区域使用当前批次所有逐帧 Mask 的并集计算，但真正送入模型和用于合成的仍是每帧自己的 Mask。并集只用于定位需要裁剪和处理的区域，不会把并集中的所有像素都当作每帧待修复像素。

各模型处理方式：

| 模型 | 精细化 Mask 行为 |
| --- | --- |
| LAMA | 每帧使用独立 Mask，批量推理时保持图像和 Mask 索引一致 |
| STTN_DET | 每帧 Mask 分别裁剪和缩放后进入时序模型 |
| ProPainter | `read_mask()` 支持 Mask 列表，光流 Mask 和修复 Mask 均按帧生成 |
| OpenCV | 每帧调用 `cv2.inpaint()` 时使用对应 Mask |

ProPainter 内部仍会执行自身的 Mask dilation，因此实际送入 ProPainter 网络的区域可能比精细化 Preview 略宽。

### 8.4 同步输出正式 Mask Preview

```ini
[integration]
write_mask_preview = true
mask_preview_output =
```

开启后，正式擦除生成 Mask 缓存时同步输出涂黑预览，不会再次执行 OCR 或再次生成 Mask。

路径留空时输出到输入视频旁：

```text
<输入文件名>_mask_preview.mp4
```

该视频使用的 Mask 与实际传给模型的最终时序 Mask 一致，并沿用 `[preview]` 中的方框显示选项。当前同步预览视频不复制原视频音轨。

## 9. OCR 区域接口

GUI 配置中的字幕区域以 `0~1` 比例保存，但 GUI 原本会在任务启动前换算为像素。为命令行增加了相同顺序的比例接口：

```text
--subtitle-area-ratios YMIN YMAX XMIN XMAX
-r YMIN YMAX XMIN XMAX
```

LAMA 示例：

```bash
python backend/main.py \
  -i input.mp4 \
  -o output.mp4 \
  --inpaint-mode lama \
  -r 0.7667 0.8944 0.0583 0.8573
```

程序打开视频并获得真实宽高后再进行换算，因此无需提前知道视频分辨率。可以重复 `-r` 指定多个区域，但比例参数不能与原像素参数 `-c` 同时使用。

## 10. Windows 路径和媒体尺寸兼容

原来的 `get_readable_path()` 在 Windows 无法取得 8.3 短路径时可能返回空字符串，导致 OpenCV 获得 0×0 视频尺寸。现在的处理方式是：

1. 输入路径先转成绝对路径；
2. 短路径转换失败时回退到正常绝对 Unicode 路径；
3. `VideoCapture` 元数据没有宽高时读取首帧获取尺寸；
4. 输入不存在、无法打开、无法解码或帧率非法时提前给出明确错误。

## 11. 配置文件分区

`subtitle_mask_config.ini` 的职责如下：

| 分区 | 用途 |
| --- | --- |
| `[integration]` | 正式擦除开关、同步 Preview 开关和输出路径 |
| `[video]` | 仅供独立 Preview 脚本使用的输入、输出和处理秒数 |
| `[ocr]` | OCR 采样、检测阈值、分区域裁剪和放大；其中 `areas/area` 仅供独立 Preview 使用 |
| `[mask]` | 主字幕颜色、边缘生长、扩框、清理和膨胀参数 |
| `[secondary_mask]` | 左侧特殊字幕的第二套颜色参数 |
| `[temporal]` | 未来帧借用、稳定性和连续性参数 |
| `[preview]` | 原始 OCR 框和扩展框显示开关 |

建议保留一份经过代表性视频验证的配置作为基线。调参时优先使用短视频和 `duration_seconds`，确认 Mask 后再运行耗时较长的正式 inpainting。

## 12. 测试覆盖

`test/subtitle_mask_checks.py` 覆盖了以下核心行为：

- Mask 只在 OCR 框及允许范围内产生；
- 宽松边缘必须连接核心像素；
- 输出为二值 `uint8`；
- 灰度约束拒绝彩色背景，但不损失核心像素；
- 主模式扩框可以连续找到相邻标点；
- 扩框不会跨过空带吸收远处白点；
- 第二模式只在左侧触发且不执行颜色扩框；
- 主模式核心像素优先于第二模式；
- Mask 稳定性、未来借用和连续性保护；
- OCR 采样框覆盖完整采样块；
- 裁剪 OCR 框坐标回映和重叠区域去重；
- 单 Mask 旧接口与逐帧 Mask 新接口兼容；
- PNG Mask 缓存往返一致；
- 比例 OCR 区域换算、范围检查和多区域；
- 同步 Preview 默认文件名生成。

当前开发环境中的原 Python 3.12 虚拟环境解释器已经缺失，系统 Python 又未安装项目所需的 NumPy/OpenCV，因此目前主要完成了静态编译、配置解析和差异检查。恢复可用虚拟环境后，应运行：

```bash
python test/subtitle_mask_checks.py
```

并使用一段短视频分别执行 LAMA、STTN_DET 和 ProPainter 集成测试。

## 13. 当前边界与后续建议

1. 精细化 Mask 依赖字幕颜色相对稳定；背景中大面积相似颜色仍可能造成误选。
2. 第二字幕模式不扩框，降低了灰色背景误选风险，但 OCR 框外的灰色标点仍可能漏掉。
3. `future_mask_frames` 太大会提前带入下一句字幕；太小则无法覆盖较慢的逐字显示。
4. OCR 框完全漏检时，单帧颜色算法没有搜索起点，只能依赖其他帧 OCR 框传播或未来 Mask。
5. 正式流程比原流程多一次视频解码和逐帧颜色计算，但无需额外执行 OCR。
6. 同步 Preview 使用 MP4 视频输出，目前不包含音轨。
7. 后续如果继续优化，可以考虑：
   - 将独立 Preview 与正式运行时的配置解析和时序函数进一步合并，减少重复代码；
   - 为 Mask 缓存增加磁盘缓存和配置哈希，避免同一视频重复生成；
   - 增加场景切换边界保护，防止未来 Mask 跨场景借用；
   - 为第二模式增加更严格的局部结构或纹理判断，而不是直接恢复颜色扩框；
   - 建立固定短视频回归集，分别统计漏检、误检、残边和时序闪烁。
