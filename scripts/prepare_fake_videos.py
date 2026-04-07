import os
import shutil

source_root = 'dataset/fake'
target_root = 'dataset/fake2'

fake_folder = [
    "DeepFakeDetection",
    "Deepfakes",
    "FaceSwap",
    "Face2Face",
    "FaceShifter",
    "NeuralTextures"
]
os.makedirs(target_root, exist_ok=True)
for folder in fake_folder:
    folder_path = os.path.join(source_root, folder) # dataset/fake/DeepFakeDetection
    videos = os.listdir(folder_path)
    for vid in videos:
        src = os.path.join(folder_path, vid) # dataset/fake/DeepFakeDetection/video1.mp4
        new_name = folder + '_' + vid # DeepFakeDetection_video1.mp4
        dst = os.path.join(target_root, new_name) # dataset/fake2/DeepFakeDetection_video1.mp4
        shutil.copy(src, dst)
        os.remove(src)
print("Done!")