import os
import sys
import logging
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
logger = logging.getLogger('train_logger')
logger.setLevel(logging.DEBUG)

c_handler = logging.StreamHandler(sys.stdout)
f_handler = logging.FileHandler('train.log', mode='w')

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(formatter)
f_handler.setFormatter(formatter)

logger.addHandler(c_handler)
logger.addHandler(f_handler)

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
CLASSES = ['__background__', 'paper', 'non-paper']
NUM_CLASSES = len(CLASSES)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class WasteDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.images_dir = os.path.join(root, 'images')
        self.labels_dir = os.path.join(root, 'labels')

        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not os.path.exists(self.labels_dir):
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.lower().endswith('.jpg')])
        logger.info(f"Initialized dataset with {len(self.image_files)} images from {self.images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, os.path.splitext(img_name)[0] + '.txt')

        # Load image
        img = Image.open(img_path).convert("RGB")
        img_width, img_height = img.size

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    try:
                        class_id, cx, cy, w, h = map(float, line.strip().split())
                        class_id = int(class_id) + 1  # background = 0

                        # Convert YOLO format to [xmin, ymin, xmax, ymax]
                        x_center, y_center = cx * img_width, cy * img_height
                        bw, bh = w * img_width, h * img_height
                        xmin = x_center - bw / 2
                        ymin = y_center - bh / 2
                        xmax = x_center + bw / 2
                        ymax = y_center + bh / 2

                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(class_id)
                    except Exception as e:
                        logger.error(f"Error parsing label in {label_path}: {e}")
        else:
            logger.warning(f"Label file not found for image {img_name}")

        # Ensure non-empty targets
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    logger.info("Loaded pre-trained Faster R-CNN (MobileNetV3 backbone).")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    logger.info(f"Modified predictor head for {num_classes} classes.")
    return model


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
def train(filename, path):
    num_epochs = 50
    batch_size = 3
    learning_rate = 0.005
    momentum = 0.9
    weight_decay = 0.0005

    transform = transforms.Compose([transforms.ToTensor()])

    dataset_root = path
    dataset = WasteDataset(root=dataset_root, transforms=transform)

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda x: tuple(zip(*x))
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    model = get_model_instance_segmentation(NUM_CLASSES).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            try:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            except Exception as e:
                logger.error(f"Training error: {e}")
                continue

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            logger.debug(f"Batch loss: {losses.item():.4f}")

        lr_scheduler.step()
        avg_loss = total_loss / len(data_loader)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] - Average Loss: {avg_loss:.4f}")

    modelname = filename
    torch.save(model.state_dict(), modelname)
    logger.info(f"Model saved to {modelname}")


if __name__ == "__main__":
    modelname = "newTrain.pth"
    datasetpath = r"E:\PythonProjects\RecyclingRobot\datasets\paper-nonpaper_dataset\train"
    train(modelname, datasetpath)