import cv2
import os
from tqdm import tqdm
import random
IMG_SIZE = 256
FRAME_INTERVAL = 5
MAX_VIDEOS = 1000

def extract_frames(video_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    videos = os.listdir(video_folder)
    random.shuffle(videos)
    videos = videos[:MAX_VIDEOS]
    for vid in tqdm(videos):
        video_path = os.path.join(video_folder, vid)
        cap = cv2.VideoCapture(video_path)
        name = os.path.splitext(vid)[0]
        video_save = os.path.join(output_folder, name)
        os.makedirs(video_save, exist_ok=True)
        frame_count = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % FRAME_INTERVAL == 0:
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(os.path.join(video_save, f"{saved}.jpg"), frame)
                saved += 1
            frame_count += 1
        cap.release()

extract_frames('dataset/fake', 'frames/fake')
extract_frames('dataset/real', 'frames/real')
print("Done!")
        