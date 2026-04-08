from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Dropout, GlobalAveragePooling2D

IMG_SIZE = 224
SEQ_LEN = 20
def build_model():
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    model = Sequential()
    model.add(TimeDistributed(base_model, input_shape=(SEQ_LEN, IMG_SIZE, IMG_SIZE, 3)))
    model.add(TimeDistributed(GlobalAveragePooling2D()))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    return model