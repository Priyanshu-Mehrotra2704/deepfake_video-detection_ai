from scripts.data_generator import DataGenerator
from scripts.model import build_model

model = build_model()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
train_gen = DataGenerator("faces")
model.fit(train_gen, epochs=10)
model.save("deepfake_model.h5")