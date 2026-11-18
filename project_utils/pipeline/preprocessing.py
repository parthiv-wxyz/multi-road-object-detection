"""
Adaptive preprocessing modules for multi-road scene detection.

These modules run automatically inside the inference pipeline without requiring
model retraining:
1. AdaptiveIlluminationNormalizer: Auto-corrects very dark (night/shadow) or
   overexposed images via gentle gamma adjustment.
2. RoadSurfaceEnhancer: Applies LAB-space CLAHE blended with the original image
   exclusively for road-damage detectors (RDD4D / RDD_22) to highlight hairline
   asphalt cracks and pothole edges without distorting vehicle/sign colors.
"""

import cv2
import numpy as np
from PIL import Image


class AdaptiveIlluminationNormalizer:
    """
    Automatically adjusts extreme low-light or overexposed frames using
    bounded gamma correction while leaving well-lit images untouched.
    """

    def __init__(self, low_thresh: float = 75.0, high_thresh: float = 210.0):
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh

    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = np.asarray(image)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mean_v = float(np.mean(hsv[:, :, 2]))

        if mean_v < self.low_thresh:
            # Brighten shadowed / dusk scenes gently
            gamma = max(0.55, min(0.85, mean_v / 110.0))
        elif mean_v > self.high_thresh:
            # Tone down washed-out bright glare
            gamma = 1.25
        else:
            return image

        lut = np.array(
            [((i / 255.0) ** gamma) * 255.0 for i in range(256)],
            dtype=np.uint8,
        )
        corrected = cv2.LUT(rgb, lut)
        return Image.fromarray(corrected)


class RoadSurfaceEnhancer:
    """
    Enhances asphalt crack and pothole visibility using LAB-space CLAHE
    blended with the original frame so textures pop without introducing
    harsh artifacts.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple = (8, 8),
        blend_alpha: float = 0.60,
    ):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.blend_alpha = blend_alpha

    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = np.asarray(image)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )
        l_enhanced = clahe.apply(l_chan)

        lab_enhanced = cv2.merge((l_enhanced, a_chan, b_chan))
        rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

        blended = cv2.addWeighted(
            rgb_enhanced,
            self.blend_alpha,
            rgb,
            1.0 - self.blend_alpha,
            0,
        )
        return Image.fromarray(blended)
