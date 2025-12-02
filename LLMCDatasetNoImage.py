import os
import csv
import cv2
from ultralytics import YOLO
import google.generativeai as genai

# ----------------------------
# 1) Paths and config
# ----------------------------

MODEL_PATH = "/Users/beyzayildirim/Downloads/best.pt"
DATASET_DIR = "/Users/beyzayildirim/Desktop/preDataset"
OUTPUT_CSV = "yolo_llm_results_last.csv"


# ----------------------------
# 2) Load YOLO model
# ----------------------------
yolo = YOLO(MODEL_PATH)

# ----------------------------
# 3) Configure Google Gemini
# ----------------------------
genai.configure(api_key="AIzaSyC4PrlrVPV5X9euK1LKSvep8RdOVXQnKzI")
model = genai.GenerativeModel("models/gemini-2.5-pro")

# ----------------------------
# 4) Function: Classify crop using LLM
# ----------------------------
def classify_crop_with_llm(crop_bgr):
    _, buffer = cv2.imencode(".jpg", crop_bgr)
    img_bytes = buffer.tobytes()

    prompt = """
    You are an image classifier for recycling.
    Classify ONLY the material of the object in this image crop.
    Valid labels:
    - plastic
    - metal
    - glass
    - paper
    Return only one word: plastic, metal, glass or paper.
    """

    response = model.generate_content(
        [
            {
                "mime_type": "image/jpeg",
                "data": img_bytes,
            },
            prompt,
        ]
    )

    return response.text.strip().lower()


# ----------------------------
# 5) Dataset loop: YOLO + LLM
# ----------------------------
def run_on_dataset():

    rows = []
    image_files = [
        f for f in os.listdir(DATASET_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for img_name in image_files:
        img_path = os.path.join(DATASET_DIR, img_name)
        print(f"Processing {img_path} ...")

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Could not read {img_path}, skipping.")
            continue

        results = yolo(frame)[0]

        if len(results.boxes) == 0:
            print("No detections.")
            continue

        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            label = classify_crop_with_llm(crop)

            yolo_cls = int(box.cls[0]) if box.cls is not None else -1
            conf = float(box.conf[0]) if box.conf is not None else -1.0

            print(f"  Box {i}: LLM={label}, YOLO_cls={yolo_cls}, conf={conf:.2f}")

            rows.append([
                img_name,
                i,
                x1, y1, x2, y2,
                yolo_cls,
                conf,
                label
            ])

    # ---- CSV’ye yaz ----
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "box_index",
            "x1", "y1", "x2", "y2",
            "yolo_class", "yolo_conf",
            "llm_label"
        ])
        writer.writerows(rows)

    print(f"\nFinished. Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_on_dataset()