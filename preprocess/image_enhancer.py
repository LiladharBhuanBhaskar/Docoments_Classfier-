import cv2
import numpy as np

def enhance_for_ocr(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not found")

    # 1️⃣ Resize (OCR loves bigger text)
    img = cv2.resize(
        img, None,
        fx=2.0, fy=2.0,
        interpolation=cv2.INTER_CUBIC
    )

    # 2️⃣ Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3️⃣ CLAHE (local contrast boost — VERY important)
    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )
    gray = clahe.apply(gray)

    # 4️⃣ Mild denoising (don’t overdo)
    gray = cv2.fastNlMeansDenoising(
        gray,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # 5️⃣ Sharpen (edge-preserving)
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharp = cv2.filter2D(gray, -1, kernel)

    # 6️⃣ Convert back to 3-channel (Azure expects color)
    final = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

    return final
