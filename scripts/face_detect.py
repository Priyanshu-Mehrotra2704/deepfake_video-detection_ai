import os
import cv2
from tqdm import tqdm
frames_dir = 'frames/real'
faces_dir = 'faces/real'
os.makedirs(faces_dir, exist_ok=True)
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# for video_folder in tqdm(os.listdir(frames_dir)):
#     video_path = os.path.join(frames_dir, video_folder)
#     if not os.path.isdir(video_path):
#         continue
#     save_folder = os.path.join(faces_dir, video_folder)
#     os.makedirs(save_folder, exist_ok=True)
#     for img in os.listdir(video_path):
#         img_path = os.path.join(video_path,img)
#         frame = cv2.imread(img_path)
#         if frame is None:
#             continue
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
#         for (x,y,w,h) in faces:
#             face = frame[y:y+h, x:x+w]
#             save_path = os.path.join(save_folder, img)
#             cv2.imwrite(save_path, face)

import os
import cv2
import tqdm

frames_root = "frames"
faces_root = "faces"

os.makedirs(faces_root, exist_ok=True)

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)


for label in ["real"]:

    frames_dir = os.path.join(frames_root, label)
    faces_dir = os.path.join(faces_root, label)

    os.makedirs(faces_dir, exist_ok=True)

    for video_folder in tqdm.tqdm(os.listdir(frames_dir)):

        video_path = os.path.join(frames_dir, video_folder)
        save_path = os.path.join(faces_dir, video_folder)

        os.makedirs(save_path, exist_ok=True)

        for img_name in os.listdir(video_path):

            img_path = os.path.join(video_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5
            )

            for i, (x, y, w, h) in enumerate(faces):

                face = img[y:y+h, x:x+w]

                face_name = img_name + "_" + str(i) + ".jpg"

                cv2.imwrite(
                    os.path.join(save_path, face_name),
                    face
                )
print("Done!")
