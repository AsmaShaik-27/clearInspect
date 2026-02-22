import os
import shutil
import random

# =========================
# CONFIG
# =========================
RAW_PATH = "data/raw"
OUTPUT_PATH = "data/processed/classification"
SPLIT_RATIO = 0.7
SEED = 42

random.seed(SEED)

CLASSES = ["good", "scratch", "hole", "liquid", "color", "combined"]

# =========================
# CREATE OUTPUT FOLDERS
# =========================
for split in ["train", "test"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_PATH, split, cls), exist_ok=True)

# =========================
# SPLIT FUNCTION
# =========================
def split_and_copy(images, class_name):
    random.shuffle(images)
    split_index = int(len(images) * SPLIT_RATIO)

    train_imgs = images[:split_index]
    test_imgs = images[split_index:]

    for img in train_imgs:
        dst = os.path.join(OUTPUT_PATH, "train", class_name, os.path.basename(img))
        if not os.path.exists(dst):
            shutil.copy2(img, dst)

    for img in test_imgs:
        dst = os.path.join(OUTPUT_PATH, "test", class_name, os.path.basename(img))
        if not os.path.exists(dst):
            shutil.copy2(img, dst)


# =========================
# GOOD CLASS (train + test good)
# =========================
good_images = []

for folder in [
    os.path.join(RAW_PATH, "train", "good"),
    os.path.join(RAW_PATH, "test", "good")
]:
    for img in os.listdir(folder):
        good_images.append(os.path.join(folder, img))

print("Total good images:", len(good_images))
split_and_copy(good_images, "good")


# =========================
# DEFECT CLASSES
# =========================
for cls in CLASSES:
    if cls == "good":
        continue

    class_path = os.path.join(RAW_PATH, "test", cls)

    images = [
        os.path.join(class_path, img)
        for img in os.listdir(class_path)
    ]

    print(f"Total {cls} images:", len(images))
    split_and_copy(images, cls)

print("Dataset successfully created at:", OUTPUT_PATH)