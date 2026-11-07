"""Document binarization operations. A True pixel represents ink."""

import cv2
import numpy as np


def read_gray(path):
    """Read a grayscale image, including paths with Chinese characters."""
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def otsu(gray):
    """Use the global threshold that maximizes between-class variance."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binary != 0


def odd_window(gray, fraction):
    """Express window widths relative to the shorter image dimension."""
    return max(3, 2 * round(min(gray.shape) * fraction / 2) + 1)


def adaptive(gray, fraction, offset):
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        odd_window(gray, fraction), offset,
    )
    return binary != 0


def correct_background(gray, fraction):
    """Estimate bright paper under dark text, then divide out its intensity."""
    width = odd_window(gray, fraction)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, width))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    corrected = cv2.divide(gray, np.maximum(background, 1), scale=255)
    return corrected, background


def median_otsu(corrected):
    return otsu(cv2.medianBlur(corrected, 3))


def sauvola(gray, fraction, weight):
    """Choose a threshold from the local mean and standard deviation."""
    width = odd_window(gray, fraction)
    image = gray.astype(np.float32)
    window = (width, width)
    mean = cv2.boxFilter(image, -1, window)
    square_mean = cv2.sqrBoxFilter(image, cv2.CV_32F, window)
    deviation = np.sqrt(np.maximum(square_mean - mean * mean, 0))
    threshold = mean * (1 + weight * (deviation / 128 - 1))
    return image < threshold


def suppress_border_strips(ink, margin, minimum_span):
    """Remove long, thin components confined to and touching an image edge."""
    height, width = ink.shape
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8))
    x, y, component_width, component_height, _ = stats.T
    side = ((x <= 0.005 * width) & (x + component_width <= margin * width)
            | (x >= (1 - margin) * width) & (x + component_width >= 0.995 * width))
    end = ((y <= 0.005 * height) & (y + component_height <= margin * height)
           | (y >= (1 - margin) * height) & (y + component_height >= 0.995 * height))
    vertical = (
        side & (component_height >= minimum_span * height)
        & (component_height >= 4 * component_width)
    )
    horizontal = (
        end & (component_width >= minimum_span * width)
        & (component_width >= 4 * component_height)
    )
    remove = vertical | horizontal
    remove[0] = False
    return ink & ~remove[labels]


def pixel_metrics(prediction, truth):
    """Score the ink class. A jointly empty mask is a perfect match."""
    if prediction.shape != truth.shape:
        raise ValueError(f"Mask shape mismatch: {prediction.shape} vs {truth.shape}")
    true_positive = int(np.count_nonzero(prediction & truth))
    false_positive = int(np.count_nonzero(prediction & ~truth))
    false_negative = int(np.count_nonzero(~prediction & truth))
    predicted = true_positive + false_positive
    actual = true_positive + false_negative
    return {
        "precision": true_positive / predicted if predicted else float(actual == 0),
        "recall": true_positive / actual if actual else float(predicted == 0),
        "f1": 2 * true_positive / (predicted + actual) if predicted + actual else 1.0,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
    }
