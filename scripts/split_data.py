import os
import shutil
import random

source = "faces"
train_dir = "faces/train"
val_dir = "faces/val"

split_ratio = 0.8

for cls in ["real", "fake"]:

    src_path = os.path.join(source, cls)

    videos = os.listdir(src_path)
    random.shuffle(videos)

    split = int(len(videos) * split_ratio)

    train_videos = videos[:split]
    val_videos = videos[split:]

    for v in train_videos:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        shutil.copytree(
            os.path.join(src_path, v),
            os.path.join(train_dir, cls, v),
            dirs_exist_ok=True
        )

    for v in val_videos:
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        shutil.copytree(
            os.path.join(src_path, v),
            os.path.join(val_dir, cls, v),
            dirs_exist_ok=True
        )

print("Train/Val split done")