import os
import numpy as np
import cv2
import random
from tensorflow.keras.utils import Sequence

IMG_SIZE = 224
SEQ_LEN = 20
BATCH_SIZE = 4   # reduce to 2 if RAM issue


class DataGenerator(Sequence):

    def __init__(self, data_path):

        self.samples = []

        for label, cls in enumerate(["real", "fake"]):

            class_path = os.path.join(data_path, cls)

            print(f"Loading from: {class_path}")

            if not os.path.exists(class_path):
                print(f"Missing folder: {class_path}")
                continue

            for vid in os.listdir(class_path):

                folder = os.path.join(class_path, vid)

                if not os.path.isdir(folder):
                    continue

                frames = os.listdir(folder)

                if len(frames) >= SEQ_LEN:
                    self.samples.append((folder, label))

        print(f"Total samples loaded: {len(self.samples)}")

        self.on_epoch_end()

    def __len__(self):
        return max(1, len(self.samples) // BATCH_SIZE)

    def on_epoch_end(self):
        random.shuffle(self.samples)

    def __getitem__(self, index):

        X, y = [], []

        while len(X) < BATCH_SIZE:

            folder, label = random.choice(self.samples)

            try:
                frames = sorted(
                    os.listdir(folder),
                    key=lambda x: int(os.path.splitext(x)[0])
                )
            except:
                continue

            if len(frames) < SEQ_LEN:
                continue

            start = random.randint(0, len(frames) - SEQ_LEN)

            seq = []

            for i in range(start, start + SEQ_LEN):

                img_path = os.path.join(folder, frames[i])

                img = cv2.imread(img_path)

                if img is None:
                    continue

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0

                seq.append(img)

            if len(seq) == SEQ_LEN:
                X.append(seq)
                y.append(label)

        return np.array(X), np.array(y)
