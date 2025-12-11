import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import random


def extract_frames(video_path):
    """
    Assumes video_path is always valid and readable. 
    If reading fails, an exception will be raised by cv2 or numpy.
    """
    video = cv2.VideoCapture(video_path)
    frames_list = []
    
    if not video.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    while True:
        success, frame = video.read()
        if not success:
            break
        
        frame = cv2.resize(frame, (256, 256))
        frames_list.append(frame)
    video.release()

    if len(frames_list) == 0:
        raise ValueError(f"No frames found in video: {video_path}")

    frames = np.stack(frames_list, axis=0)
    frames = np.transpose(frames, (0, 3, 1, 2))
    return frames


class MSADataset(Dataset):
    def __init__(self, data_dir, pose_dir, audio_dir, is09_dir, batch_size, phase):
        self.data = []
        self.pose_dir = pose_dir
        self.audio_dir = audio_dir
        self.is09_dir = is09_dir
        self.phase = phase
        self.batch_size = batch_size
        phase_dir = os.path.join(data_dir, phase)
        print(f"Loading data from: {phase_dir}")
        if not os.path.exists(phase_dir):
            raise ValueError(f"Data directory does not exist: {phase_dir}")
        
        for root, _, files in os.walk(phase_dir):
            for file in files:
                if file.endswith('.mp4'):
                    self.data.append(os.path.join(root, file))
        
        print(f"Number of samples in dataset: {len(self.data)}")
        if not self.data:
            raise ValueError(f"No .mp4 files found in {phase_dir}")

    def __getitem__(self, index):
        video_path = self.data[index]
        frames = extract_frames(video_path)
        frames = torch.from_numpy(frames).float().cpu()
        base_name = os.path.basename(video_path)
        label_part = base_name.split('_')[-1].split('.')[0]
        label = int(label_part)

    
        pose_base_name = base_name.split('.')[0]
        pose_file = f"POSE{pose_base_name}.npy"
        pose_path = os.path.join(self.pose_dir, self.phase, pose_file)
        
        pose_features = torch.from_numpy(np.load(pose_path)).float().cpu()
        
     
        max_people = 8
        current_people = pose_features.size(0)
        if current_people < max_people:
            padding_size = max_people - current_people
            padding_tensor = torch.zeros(padding_size, pose_features.size(1), pose_features.size(2),
                                         pose_features.size(3), device='cpu')
            pose_features = torch.cat((pose_features.cpu(), padding_tensor), dim=0)
        else:
            pose_features = pose_features[:max_people]

      
        audio_file = f"{base_name.split('.')[0]}.npz"
        audio_path = os.path.join(self.audio_dir, self.phase, audio_file)
        audio_data = np.load(audio_path)
        audio_features = torch.from_numpy(audio_data['feat']).float().cpu()

   
        is09_file = f"IS09{base_name.split('.')[0]}.npy"
        is09_path = os.path.join(self.is09_dir, self.phase, is09_file)
        is09_features = torch.from_numpy(np.load(is09_path)).float().cpu()

        return frames, pose_features, audio_features, is09_features, label

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    
    frames, poses, audios, is09s, labels = zip(*batch)


    target_frame_length = 150
    frames_padded = []
    for frame in frames:
        frame = frame.cpu()
        if frame.size(0) < target_frame_length:
            padding_size = target_frame_length - frame.size(0)
            padding_tensor = torch.zeros(padding_size, *frame.size()[1:], device='cpu')
            frame = torch.cat((frame, padding_tensor), dim=0)
        else:
            frame = frame[:target_frame_length]
        frames_padded.append(frame)
    frames_padded = torch.stack(frames_padded).cpu()


    processed_poses = []
    for pose in poses:
        pose = pose.cpu()
        if pose.size(1) < 150:
            padding_size = 150 - pose.size(1)
            padding_tensor = torch.zeros(pose.size(0), padding_size, pose.size(2), pose.size(3), device='cpu')
            pose = torch.cat((pose, padding_tensor), dim=1)
        else:
            pose = pose[:, :150, :, :]
        processed_poses.append(pose)
    poses_padded = pad_sequence(processed_poses, batch_first=True).cpu()


    audios_padded = []
    target_audio_length = 150
    for audio in audios:
        audio = audio.cpu()
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)
        if audio.size(0) < target_audio_length:
            padding_size = target_audio_length - audio.size(0)
            padding_tensor = torch.zeros(padding_size, audio.size(1), audio.size(2), device='cpu')
            audio = torch.cat((audio, padding_tensor), dim=0)
        else:
            audio = audio[:target_audio_length, :, :]
        audios_padded.append(audio)
    audios_padded = torch.stack(audios_padded).cpu()

   
    is09s_stacked = torch.stack(is09s).cpu()


    labels_tensor = torch.tensor(labels, dtype=torch.long, device='cpu')

    return frames_padded, poses_padded, audios_padded, is09s_stacked, labels_tensor


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32) + worker_id
    np.random.seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)


def get_loader(data_dir, pose_dir, audio_dir, is09_dir, batch_size, phase='train', shuffle=True, generator=None):
    dataset = MSADataset(data_dir, pose_dir, audio_dir, is09_dir, batch_size, phase)
    print(f"Data loader phase: {phase}, number of samples: {len(dataset)}")

    if generator is None:
        generator_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        generator = torch.Generator(device=generator_device)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator
    )

    return data_loader