"""A5 train-time augmentation (D-021): OpenCV, applied to unpadded TRAIN crops before padding.
Val and test crops are never augmented."""
import cv2

FLIP_P = 0.5                       # horizontal flip probability
BRIGHTNESS_RANGE = (0.8, 1.2)      # brightness multiplier ~ Uniform[0.8, 1.2]


def hflip(img):
    """Mirror left to right."""
    return cv2.flip(img, 1)


def adjust_brightness(img, factor):
    """Multiply every pixel by `factor`; OpenCV saturates the result to 0..255 (uint8)."""
    return cv2.convertScaleAbs(img, alpha=float(factor), beta=0)


def augment_train(img, rng, flip_p=FLIP_P, brightness_range=BRIGHTNESS_RANGE):
    """Random flip then brightness. `rng` is a numpy Generator. Returns (image, what_was_applied)."""
    flipped = bool(rng.random() < flip_p)
    if flipped:
        img = hflip(img)
    factor = float(rng.uniform(*brightness_range))
    return adjust_brightness(img, factor), {"flipped": flipped, "brightness": factor}
