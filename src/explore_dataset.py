from pathlib import Path

ROOT = Path("data/raw")

def is_img(p: Path) -> bool:
    return p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

print("ROOT exists:", ROOT.exists())
print("Top-level items:", [x.name for x in ROOT.iterdir()] if ROOT.exists() else [])

# Try to find train/test folders anywhere under data/raw
train_dirs = [p for p in ROOT.rglob("*") if p.is_dir() and p.name.lower() == "train"]
test_dirs  = [p for p in ROOT.rglob("*") if p.is_dir() and p.name.lower() == "test"]

print("\nTrain dirs found:", train_dirs[:5])
print("Test dirs found :", test_dirs[:5])

def count_split(split_dir: Path):
    classes = [d for d in split_dir.iterdir() if d.is_dir()]
    counts = {}
    for c in classes:
        n = sum(1 for f in c.rglob("*") if f.is_file() and is_img(f))
        counts[c.name] = n
    return counts

if train_dirs:
    tr = train_dirs[0]
    print("\nUsing train:", tr)
    print("Train class counts:", count_split(tr))

if test_dirs:
    te = test_dirs[0]
    print("\nUsing test:", te)
    print("Test class counts:", count_split(te))
