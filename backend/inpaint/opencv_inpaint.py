import cv2

from backend.tools.inpaint_tools import normalize_frame_masks

class OpenCVInpaint:

    def __init__(self):
        pass

    def inpaint(self, frame, mask):
        return cv2.inpaint(frame, mask, 3, cv2.INTER_LINEAR)

    def __call__(self, frames, mask):
        masks = normalize_frame_masks(mask, len(frames))
        return [self.inpaint(frame, masks[index]) for index, frame in enumerate(frames)]
