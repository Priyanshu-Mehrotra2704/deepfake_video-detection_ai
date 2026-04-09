from scripts.data_generator import DataGenerator
from scripts.model import build_model

model = build_model()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

train_gen = DataGenerator("faces/train")
val_gen = DataGenerator("faces/val")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5
)

model.save("deepfake_model.h5")