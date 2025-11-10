import cv2
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import easyocr
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
MODEL_DIR = "./text_classifier/checkpoint-100"  # path to your fine-tuned model
CONFIDENCE_THRESHOLD = 0.5     # minimum OCR confidence
CAM_INDEX = 0                  # change if you use external webcam

# -----------------------------
# LOAD OCR + LLM CLASSIFIER
# -----------------------------
print("Loading OCR model...")
reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

print("Loading LLM classifier...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Retrieve class labels
id2label = model.config.id2label
if not id2label:
    id2label = {0: "class_0", 1: "class_1"}  # fallback

# -----------------------------
# TEXT CLASSIFICATION FUNCTION
# -----------------------------
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        pred_label = id2label.get(pred_id, "unknown")
        confidence = probs[0][pred_id].item()
    return pred_label, confidence

# -----------------------------
# MAIN LOOP
# -----------------------------
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("❌ Could not open camera.")
        return

    print("✅ Camera started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for speed
        resized = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # OCR detection
        results = reader.readtext(rgb_frame)

        for (bbox, text, conf) in results:
            if conf < CONFIDENCE_THRESHOLD:
                continue

            # Draw bounding box
            pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
            cv2.polylines(resized, [pts], True, (0, 255, 0), 2)

            # Classify extracted text
            pred_label, pred_conf = classify_text(text)

            # Draw text and class
            label_str = f"{text} → {pred_label} ({pred_conf:.2f})"
            cv2.putText(resized, label_str, (int(bbox[0][0]), int(bbox[0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Show frame
        cv2.imshow("Live OCR + Classifier", resized)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
