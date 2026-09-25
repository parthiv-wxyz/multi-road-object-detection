import torch


class WiseIoU:
    """
    Wise-IoU v3 loss for YOLOv5 xywh bounding boxes.

    Input format:
        [x_center, y_center, width, height]

    Returns:
        Per-box Wise-IoU v3 loss.
    """

    def __init__(self, alpha=1.9, delta=3.0, momentum=0.01):
        self.alpha = alpha
        self.delta = delta
        self.momentum = momentum

        self.running_mean = None

    def __call__(self, pred, target):
        """
        pred:
            [N, 4] predicted boxes in xywh

        target:
            [N, 4] target boxes in xywh
        """

        if pred.numel() == 0:
            return pred.new_zeros(())

        # --------------------------------------------------
        # xywh -> xyxy
        # --------------------------------------------------

        px, py, pw, ph = pred.unbind(1)
        tx, ty, tw, th = target.unbind(1)

        p_x1 = px - pw / 2
        p_y1 = py - ph / 2
        p_x2 = px + pw / 2
        p_y2 = py + ph / 2

        t_x1 = tx - tw / 2
        t_y1 = ty - th / 2
        t_x2 = tx + tw / 2
        t_y2 = ty + th / 2

        # --------------------------------------------------
        # Intersection
        # --------------------------------------------------

        ix1 = torch.maximum(p_x1, t_x1)
        iy1 = torch.maximum(p_y1, t_y1)
        ix2 = torch.minimum(p_x2, t_x2)
        iy2 = torch.minimum(p_y2, t_y2)

        iw = (ix2 - ix1).clamp(min=0)
        ih = (iy2 - iy1).clamp(min=0)

        intersection = iw * ih

        # --------------------------------------------------
        # IoU
        # --------------------------------------------------

        pred_area = pw.clamp(min=1e-7) * ph.clamp(min=1e-7)
        target_area = tw.clamp(min=1e-7) * th.clamp(min=1e-7)

        union = pred_area + target_area - intersection

        iou = intersection / (union + 1e-7)
        iou = iou.clamp(0.0, 1.0)

        iou_loss = 1.0 - iou

        # --------------------------------------------------
        # Center distance
        # --------------------------------------------------

        center_distance = (
            (px - tx).pow(2)
            + (py - ty).pow(2)
        )

        # --------------------------------------------------
        # Smallest enclosing box
        # --------------------------------------------------

        c_x1 = torch.minimum(p_x1, t_x1)
        c_y1 = torch.minimum(p_y1, t_y1)
        c_x2 = torch.maximum(p_x2, t_x2)
        c_y2 = torch.maximum(p_y2, t_y2)

        cw = (c_x2 - c_x1).clamp(min=1e-7)
        ch = (c_y2 - c_y1).clamp(min=1e-7)

        convex_diagonal = (
            cw.detach().pow(2)
            + ch.detach().pow(2)
        )

        # --------------------------------------------------
        # WIoU v1 distance attention
        # --------------------------------------------------

        r_wiou = torch.exp(
            center_distance /
            convex_diagonal.clamp(min=1e-7)
        )

        wiou_v1 = r_wiou * iou_loss

        # --------------------------------------------------
        # Running mean
        # --------------------------------------------------

        current_mean = iou_loss.detach().mean()

        if self.running_mean is None:
            self.running_mean = current_mean
        else:
            self.running_mean = (
                (1.0 - self.momentum) * self.running_mean
                + self.momentum * current_mean
            )

        # --------------------------------------------------
        # Outlier degree
        # --------------------------------------------------

        beta = (
            self.running_mean.detach()
            / iou_loss.detach().clamp(min=1e-7)
        )

        # --------------------------------------------------
        # WIoU v3 dynamic focusing
        # --------------------------------------------------

        alpha = torch.tensor(
            self.alpha,
            device=beta.device,
            dtype=beta.dtype,
        )

        r = beta / (
            self.delta *
            torch.pow(alpha, beta - self.delta)
        )

        # --------------------------------------------------
        # Final WIoU v3
        # --------------------------------------------------

        loss = r * wiou_v1

        return loss