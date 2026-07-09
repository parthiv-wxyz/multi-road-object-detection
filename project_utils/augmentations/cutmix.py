"""
CutMix augmentation for YOLOv5

Paper:
Enhanced YOLOv5s Model for Improved Multi-Sized Object Detection
"""

import random
import numpy as np
import cv2


class CutMix:
    def __init__(self, probability=0.5, beta=1.0):
        self.probability = probability
        self.beta = beta

    def rand_bbox(self, width, height, lam):
        cut_ratio = np.sqrt(1.0 - lam)

        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)

        cx = np.random.randint(width)
        cy = np.random.randint(height)

        x1 = np.clip(cx - cut_w // 2, 0, width)
        y1 = np.clip(cy - cut_h // 2, 0, height)
        x2 = np.clip(cx + cut_w // 2, 0, width)
        y2 = np.clip(cy + cut_h // 2, 0, height)

        return x1, y1, x2, y2

    def __call__(self, img1, labels1, img2, labels2):

        if random.random() > self.probability:
            return img1, labels1

        h, w = img1.shape[:2]

        lam = np.random.beta(self.beta, self.beta)

        x1, y1, x2, y2 = self.rand_bbox(w, h, lam)

        mixed = img1.copy()
        mixed[y1:y2, x1:x2] = img2[y1:y2, x1:x2]

        labels = []

        for l in labels1:
            labels.append(l)

        for l in labels2:
            labels.append(l)

        return mixed, np.array(labels)