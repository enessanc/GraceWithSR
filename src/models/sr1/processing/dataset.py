import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch.nn.functional as F

class ProcessedVideoDataset(Dataset):
    """İşlenmiş video veri seti"""
    def __init__(self, video_paths, target_size=(64, 64), num_frames=30):
        self.video_paths = video_paths
        self.target_size = target_size
        self.num_frames = num_frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        cap = cv2.VideoCapture(video_path)

        # Video özelliklerini al
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Frame'leri eşit aralıklarla seç
        frame_indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)

        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.target_size)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            else:
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.float32))

        cap.release()

        # Frame'leri tensor'a çevir
        frames = torch.from_numpy(np.array(frames))  # [num_frames, height, width, channels]

        # Her frame'i ayrı ayrı işle
        lr_frames = []
        hr_frames = []

        for frame in frames:
            # Channel first formatına çevir
            frame = frame.permute(2, 0, 1)  # [channels, height, width]

            # Düşük çözünürlüklü versiyonu oluştur
            lr_frame = F.interpolate(frame.unsqueeze(0), scale_factor=0.25, mode='bilinear').squeeze(0)

            lr_frames.append(lr_frame)
            hr_frames.append(frame)

        # Frame'leri stack et
        lr_frames = torch.stack(lr_frames)  # [num_frames, channels, height, width]
        hr_frames = torch.stack(hr_frames)  # [num_frames, channels, height, width]

        return lr_frames, hr_frames 