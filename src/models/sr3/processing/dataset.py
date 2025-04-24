import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    """Video veri seti"""
    def __init__(self, video_paths, hr_size=(240, 240), lr_size=(128, 128), num_frames=4):
        self.video_paths = video_paths
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.num_frames = num_frames

        # Memory optimizasyonu için frame'leri önceden yükle
        self.frames_cache = {}
        self.current_video = None
        self.current_frames = None

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]

        # Eğer aynı video ise cache'den al
        if video_path == self.current_video and self.current_frames is not None:
            frames_hr, frames_lr = self.current_frames
        else:
            # Yeni video için cache'i temizle
            self.current_video = video_path
            self.current_frames = None

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)

            frames_hr = []
            frames_lr = []

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Yüksek çözünürlüklü frame
                    frame_hr = cv2.resize(frame, self.hr_size)
                    frame_hr = frame_hr.astype(np.float32) / 255.0
                    frames_hr.append(frame_hr)

                    # Düşük çözünürlüklü frame
                    frame_lr = cv2.resize(frame, self.lr_size)
                    frame_lr = frame_lr.astype(np.float32) / 255.0
                    frames_lr.append(frame_lr)

            cap.release()

            if len(frames_hr) == 0:
                frames_hr = [np.zeros((self.hr_size[1], self.hr_size[0], 3), dtype=np.float32)
                            for _ in range(self.num_frames)]
                frames_lr = [np.zeros((self.lr_size[1], self.lr_size[0], 3), dtype=np.float32)
                            for _ in range(self.num_frames)]

            # Cache'e kaydet
            self.current_frames = (frames_hr, frames_lr)

        # Tensor'a çevir ve boyutları düzenle
        frames_hr = torch.from_numpy(np.array(frames_hr, dtype=np.float32))  # (num_frames, H, W, C)
        frames_lr = torch.from_numpy(np.array(frames_lr, dtype=np.float32))

        # Permute işlemi
        frames_hr = frames_hr.permute(0, 3, 1, 2)  # (num_frames, C, H, W)
        frames_lr = frames_lr.permute(0, 3, 1, 2)

        return frames_lr, frames_hr 