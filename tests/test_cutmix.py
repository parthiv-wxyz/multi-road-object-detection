from pathlib import Path
import sys
import cv2

# Add project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.augmentations.cutmix import CutMix

# Images located in the tests folder
img1 = cv2.imread(str(Path(__file__).parent / "sample1.jpg"))
img2 = cv2.imread(str(Path(__file__).parent / "sample2.jpg"))

assert img1 is not None, "sample1.jpg not found"
assert img2 is not None, "sample2.jpg not found"

cutmix = CutMix(probability=1.0)

img, labels = cutmix(img1, [], img2, [])

cv2.imshow("CutMix", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Image 1:", img1.shape)
print("Image 2:", img2.shape)
print("Mixed   :", img.shape)

labels1 = [[0, 0.5, 0.5, 0.2, 0.2]]
labels2 = [[1, 0.3, 0.3, 0.1, 0.1]]

img, labels = cutmix(img1, labels1, img2, labels2)

print("Labels:", labels)