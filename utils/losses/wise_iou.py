import torch
import torch.nn as nn


def bbox_iou(box1,
             box2,
             xywh=True,
             eps=1e-7):
    """
    IoU calculation

    box:
    [x,y,w,h] if xywh=True
    """

    if xywh:

        x1, y1, w1, h1 = box1.unbind(-1)
        x2, y2, w2, h2 = box2.unbind(-1)

        b1_x1 = x1 - w1 / 2
        b1_y1 = y1 - h1 / 2
        b1_x2 = x1 + w1 / 2
        b1_y2 = y1 + h1 / 2

        b2_x1 = x2 - w2 / 2
        b2_y1 = y2 - h2 / 2
        b2_x2 = x2 + w2 / 2
        b2_y2 = y2 + h2 / 2

    else:

        b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)

    inter = (
        (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0)
        *
        (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    )

    area1 = (b1_x2 - b1_x1).clamp(0) * (b1_y2 - b1_y1).clamp(0)
    area2 = (b2_x2 - b2_x1).clamp(0) * (b2_y2 - b2_y1).clamp(0)

    union = area1 + area2 - inter + eps

    return inter / union


class WiseIoULoss(nn.Module):
    """
    Simplified Wise-IoU v3 implementation.
    """

    def __init__(self,
                 alpha=1.7,
                 delta=2.7):

        super().__init__()

        self.alpha = alpha
        self.delta = delta

    def forward(self,
                pred_boxes,
                target_boxes):

        iou = bbox_iou(
            pred_boxes,
            target_boxes,
            xywh=True
        )

        beta = 1.0 - iou.detach()

        weight = torch.pow(beta, self.alpha)

        loss = weight * (1.0 - iou)

        return loss.mean()