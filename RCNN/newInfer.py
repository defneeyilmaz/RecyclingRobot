import sys
import os
import logging
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
logger = logging.getLogger("infer_logger")
logger.setLevel(logging.DEBUG)

if not logger.handlers:  # Prevent duplicate handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler("infer.log", mode="w")

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
CLASSES = ['__background__', 'non-paper', 'paper']
NUM_CLASSES = len(CLASSES)


# ------------------------------------------------------------
# Model loader
# ------------------------------------------------------------
def get_model_instance_segmentation(num_classes):
    # Match training definition exactly
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    logger.info("Loaded Faster R-CNN (MobileNetV3 backbone) with pretrained weights.")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    logger.info("Adjusted classifier head for custom number of classes.")
    return model


# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------
def infer(image_path, filename):
    logger.info(f"Starting inference on: {image_path}")

    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        sys.exit(1)

    # Transform
    transform = transforms.Compose([transforms.ToTensor()])

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model = get_model_instance_segmentation(NUM_CLASSES)
    try:
        model.load_state_dict(torch.load(filename, map_location=device))
        logger.info("Model weights loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)

    model.to(device)
    model.eval()

    # Load image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"Error opening image: {e}")
        sys.exit(1)

    input_tensor = transform(image).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model([input_tensor])

    outputs = outputs[0]
    boxes = outputs.get("boxes", [])
    labels = outputs.get("labels", [])
    scores = outputs.get("scores", [])

    if len(scores) == 0:
        logger.warning("No predictions returned by the model.")
        return

    # Visualization setup
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(image)
    confidence_threshold = 0.5

    detected = 0
    for box, label, score in zip(boxes, labels, scores):
        if score < confidence_threshold:
            continue

        detected += 1
        xmin, ymin, xmax, ymax = map(float, box.tolist())
        label = int(label.item())
        class_name = CLASSES[label] if label < len(CLASSES) else f"Unknown({label})"

        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor="lime",
            facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, f"{class_name}: {score:.2f}",
                color="yellow", fontsize=10, weight="bold")

        logger.info(f"Detected {class_name} (score={score:.2f}) at [{xmin:.1f}, {ymin:.1f}, {xmax:.1f}, {ymax:.1f}]")

    if detected == 0:
        logger.warning("No objects detected above threshold.")
    else:
        logger.info(f"Total detections: {detected}")

    plt.axis("off")
    plt.tight_layout()
    plt.show()  # or save: plt.savefig('output.jpg')

    logger.info("Inference completed successfully.")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    modelname = 'newTrain.pth'
    # path = r"E:\PythonProjects\RecyclingRobot\datasets\paper-nonpaper_dataset\valid\images/"
    # img = 'rgb_image11-8-16-58-52_jpg.rf.c706e913af4f1f920e6b990aa3e9f026.jpg'
    # infer(path+img, filename)

    # Directory containing images
    image_dir = r"E:\PythonProjects\RecyclingRobot\datasets\paper-nonpaper_dataset\valid\images"

    if not os.path.exists(image_dir):
        logger.error(f"Image directory not found: {image_dir}")
        sys.exit(1)

    # Find all JPG files (case-insensitive)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]

    if not image_files:
        logger.warning(f"No .jpg images found in: {image_dir}")
        sys.exit(0)

    logger.info(f"Found {len(image_files)} images for inference in {image_dir}")

    # Loop through each image and run inference
    for filename in image_files:
        image_path = os.path.join(image_dir, filename)
        logger.info(f"Running inference on {filename}...")
        try:
            infer(image_path, modelname)
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")