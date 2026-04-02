import cv2

def detect_document_source(f_path):
    """Determines if a document is Digital (original PDF) or Scanned (Photo/OCR)."""
    if f_path.lower().endswith('.pdf'):
        # In a real scenario, you'd check for embedded text vs images in the PDF
        return "Digital (PDF Source)"
    else:
        # Check image blur/laplacian variance to identify scanned photos
        img = cv2.imread(f_path)
        if img is None:
            raise ValueError("Unable to read image for source detection.")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return "Scanned/Photo" if variance < 500 else "High-Quality Scan"

if __name__ == "__main__":
    test_path = r"C:\Users\bhask\Downloads\SK Aadhar Card Cropped.pdf"
    source = detect_document_source(test_path)
    print(f"Document Source: {source}")