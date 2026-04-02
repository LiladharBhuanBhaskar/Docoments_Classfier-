import tensorflow as tf
import numpy as np
import cv2
from keras.layers import TFSMLayer
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.savedmodel")
LABEL_PATH = os.path.join(BASE_DIR, "labels.txt")

# print("Loading model from:", MODEL_PATH)
# print("Exists:", os.path.exists(MODEL_PATH))


model = TFSMLayer(MODEL_PATH, call_endpoint="serving_default")

# Load labels
with open(LABEL_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

IMG_SIZE = 224

def predict_document(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    preds = model(img)

    # If output is dict → extract values
    if isinstance(preds, dict):
        preds = list(preds.values())[0]

    preds = preds.numpy()

    class_index = np.argmax(preds)
    confidence = float(np.max(preds))

    return labels[class_index], confidence


if __name__ == "__main__":
    img_path = r"C:\Users\bhask\OneDrive\Desktop\test\pan\e6fb4386-imgtopdf_generated_2508201303047_page-147_jpg.rf.800091c68a9c6a27cfd1eb1f2f600bc8.jpg"
    label, conf = predict_document(img_path)
    print(label, conf)
