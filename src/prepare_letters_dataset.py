# prepare_letters_dataset.py
import random, shutil
from pathlib import Path
import json, yaml

SRC_IMG = Path("datasets/images")
SRC_LBL = Path("datasets/labels")
OUT_IMG = Path("datasets/letters/images")
OUT_LBL = Path("datasets/letters/labels")
OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)
(OUT_IMG/"train").mkdir(parents=True, exist_ok=True)
(OUT_IMG/"val").mkdir(parents=True, exist_ok=True)
(OUT_LBL/"train").mkdir(parents=True, exist_ok=True)
(OUT_LBL/"val").mkdir(parents=True, exist_ok=True)

all_images = sorted([p for p in SRC_IMG.glob("*.*") if p.suffix.lower() in (".jpg",".png",".jpeg")])
random.seed(42)
random.shuffle(all_images)
val_count = max(1, int(len(all_images)*0.12))
val = set(all_images[:val_count])
train = all_images[val_count:]

def copy_set(lst, split):
    for i, img in enumerate(lst):
        lbl = SRC_LBL / (img.stem + ".txt")
        dst_img = OUT_IMG / split / img.name
        dst_lbl = OUT_LBL / split / (img.stem + ".txt")
        shutil.copy2(img, dst_img)
        if lbl.exists():
            shutil.copy2(lbl, dst_lbl)
        else:
            dst_lbl.write_text("", encoding="utf-8")

copy_set(train, "train")
copy_set(list(val), "val")

# build names list from classes.json (ordered by id)
classes = json.loads(Path("classes.json").read_text(encoding="utf-8"))
maxid = max(classes.values())
names = [None]*(maxid+1)
for k,v in classes.items():
    names[v] = k

data = {
  "train": str((OUT_IMG/"train").resolve()),
  "val":   str((OUT_IMG/"val").resolve()),
  "nc": len(names),
  "names": names
}
Path("data").mkdir(exist_ok=True)
Path("data/letters.yaml").write_text(yaml.dump(data), encoding="utf-8")
print("Prepared dataset. Train:", len(train), "Val:", len(val))
print("Wrote data/letters.yaml with nc=", data["nc"])
