from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.losses.wise_iou import WiseIoULoss

loss_fn = WiseIoULoss()

pred = torch.tensor([
    [0.5,0.5,0.4,0.4],
    [0.2,0.2,0.1,0.1]
])

target = torch.tensor([
    [0.5,0.5,0.4,0.4],
    [0.3,0.3,0.1,0.1]
])

loss = loss_fn(pred,target)

print(loss)