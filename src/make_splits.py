from pathlib import Path
import random, shutil, hashlib, os

random.seed(42)

RAW = Path("data/raw/bottle")
OUT = Path("data/processed/bottle_splits")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

good_train = RAW / "train" / "good"
test_dir   = RAW / "test"

classes = ["good", "broken_large", "broken_small", "contamination"]

def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def safe_rmtree(path: Path):
    # Windows sometimes marks files read-only; clear attrs before delete
    if not path.exists():
        return
    for p in path.rglob("*"):
        try:
            os.chmod(p, 0o777)
        except Exception:
            pass
    shutil.rmtree(path, ignore_errors=True)

# Collect file paths
per_class = {c: [] for c in classes}

# good from train/good and test/good
per_class["good"] += [p for p in good_train.rglob("*") if is_img(p)]
per_class["good"] += [p for p in (test_dir / "good").rglob("*") if is_img(p)]

# defects from test folders
for c in ["broken_large", "broken_small", "contamination"]:
    per_class[c] += [p for p in (test_dir / c).rglob("*") if is_img(p)]

# Split ratios
train_r, val_r, test_r = 0.7, 0.15, 0.15

# Clean output safely
safe_rmtree(OUT)
OUT.mkdir(parents=True, exist_ok=True)

def unique_name(src: Path, cls: str) -> str:
    # Make filename unique even if different folders have same name like 012.png
    h = hashlib.md5(str(src).encode("utf-8")).hexdigest()[:8]
    return f"{cls}_{h}{src.suffix.lower()}"

def copy_files(file_list, split, cls):
    dest = OUT / split / cls
    dest.mkdir(parents=True, exist_ok=True)
    for src in file_list:
        dst = dest / unique_name(src, cls)
        shutil.copy2(src, dst)

summary = {}
for cls in classes:
    files = per_class[cls]
    random.shuffle(files)
    n = len(files)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)

    train_files = files[:n_train]
    val_files   = files[n_train:n_train+n_val]
    test_files  = files[n_train+n_val:]

    copy_files(train_files, "train", cls)
    copy_files(val_files, "val", cls)
    copy_files(test_files, "test", cls)

    summary[cls] = (len(train_files), len(val_files), len(test_files))

print("✅ Created splits at:", OUT)
print("Class counts (train, val, test):")
for k, v in summary.items():
    print(f"{k:15s} -> {v}")
