import os
import shutil
from sklearn.model_selection import train_test_split

base = "dataset/path"
dest = "datasets/garbage"

# Create folders
for split in ["train", "test"]:
    for cls in os.listdir(base):
        os.makedirs(os.path.join(dest, split, cls), exist_ok=True)

# Split each class into train/val (80/20)
for cls in os.listdir(base):
    class_dir = os.path.join(base, cls)
    images = [os.path.join(class_dir, f) for f in os.listdir(class_dir)]
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

    for img_path in train_imgs:
        shutil.copy(img_path, os.path.join(dest, "train", cls))
    for img_path in val_imgs:
        shutil.copy(img_path, os.path.join(dest, "test", cls))