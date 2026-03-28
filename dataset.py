import os
import glob
import json
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    from .config import Config
    from .transforms import get_train_transforms, get_val_transforms, get_degradation_transforms
except ImportError:
    from config import Config
    from transforms import get_train_transforms, get_val_transforms, get_degradation_transforms


class EndToEndDataset(Dataset):
    """
    Single-frame license plate dataset for End-to-End SR + Recognition.
    
    Loads (LR, HR, Label) pairs from tracks.
    """
    
    def __init__(self, root_dir, mode='train', split_ratio=0.9):
        self.mode = mode
        self.samples = []
        
        if mode == 'train':
            self.transform_lr = get_train_transforms(Config.IMG_HEIGHT, Config.IMG_WIDTH)
            self.transform_hr = get_train_transforms(Config.HR_IMG_HEIGHT, Config.HR_IMG_WIDTH)
            self.degrade = get_degradation_transforms()
        else:
            self.transform_lr = get_val_transforms(Config.IMG_HEIGHT, Config.IMG_WIDTH)
            self.transform_hr = get_val_transforms(Config.HR_IMG_HEIGHT, Config.HR_IMG_WIDTH)
            self.degrade = None

        print(f"[{mode.upper()}] Scanning: {root_dir}")
        abs_root = os.path.abspath(root_dir)
        search_path = os.path.join(abs_root, "**", "track_*")
        all_tracks = sorted(glob.glob(search_path, recursive=True))
        
        if not all_tracks:
            print("❌ LỖI: Không tìm thấy data.")
            return

        # Only split if it's training or validation on train set
        if 'test' in root_dir or mode == 'test':
            selected_tracks = all_tracks
        else:
            train_tracks, val_tracks = self._split_tracks(all_tracks, split_ratio)
            selected_tracks = train_tracks if mode == 'train' else val_tracks
        
        print(f"[{mode.upper()}] Loaded {len(selected_tracks)} tracks.")
        self._load_samples(selected_tracks)
    
    def _split_tracks(self, all_tracks, split_ratio):
        train_tracks = []
        val_tracks = []
        
        if os.path.exists(Config.VAL_SPLIT_FILE):
            print(f"📂 Loading split from '{Config.VAL_SPLIT_FILE}'...")
            try:
                with open(Config.VAL_SPLIT_FILE, 'r') as f:
                    val_ids = set(json.load(f))
            except:
                val_ids = set()
            for t in all_tracks:
                if os.path.basename(t) in val_ids:
                    val_tracks.append(t)
                else:
                    train_tracks.append(t)
            
            if not val_tracks and len(all_tracks) > 0:
                train_tracks, val_tracks = self._create_new_split(all_tracks, split_ratio)
        else:
            train_tracks, val_tracks = self._create_new_split(all_tracks, split_ratio)
        
        return train_tracks, val_tracks
    
    def _create_new_split(self, all_tracks, split_ratio):
        random.Random(Config.SEED).shuffle(all_tracks)
        split_idx = int(len(all_tracks) * split_ratio)
        train_tracks = all_tracks[:split_idx]
        val_tracks = all_tracks[split_idx:]
        
        val_ids = [os.path.basename(t) for t in val_tracks]
        with open(Config.VAL_SPLIT_FILE, 'w') as f:
            json.dump(val_ids, f, indent=2)
        
        return train_tracks, val_tracks
    
    def _load_samples(self, tracks):
        for track_path in tqdm(tracks, desc=f"Indexing {self.mode}"):
            json_path = os.path.join(track_path, "annotations.json")
            label = ""
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        data = data[0]
                    label = data.get('plate_text', data.get('license_plate', data.get('text', '')))
                except:
                    pass
            
            # If training, we MUST have a label
            if self.mode == 'train' and not label:
                continue

            lr_files = sorted(glob.glob(os.path.join(track_path, "lr-*.png")) + glob.glob(os.path.join(track_path, "lr-*.jpg")))
            hr_files = sorted(glob.glob(os.path.join(track_path, "hr-*.png")) + glob.glob(os.path.join(track_path, "hr-*.jpg")))
            
            if not lr_files and not hr_files:
                continue

            # Pair them up
            for i in range(max(len(lr_files), len(hr_files))):
                lr_path = lr_files[i % len(lr_files)] if lr_files else None
                hr_path = hr_files[i % len(hr_files)] if hr_files else None
                
                if lr_path or hr_path:
                    self.samples.append({
                        'lr_path': lr_path,
                        'hr_path': hr_path,
                        'label': label,
                        'track_id': os.path.basename(track_path)
                    })

    def __len__(self):
        return len(self.samples)

    def _read_image(self, path):
        if not path or not os.path.exists(path):
            return None
        image = cv2.imread(path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, idx):
        item = self.samples[idx]
        label = item['label']
        
        lr_img = self._read_image(item['lr_path'])
        hr_img = self._read_image(item['hr_path'])

        # Fallbacks if one is missing
        if lr_img is None and hr_img is not None:
            lr_img = cv2.resize(hr_img, (Config.IMG_WIDTH, Config.IMG_HEIGHT))
            if self.mode == 'train' and self.degrade:
                lr_img = self.degrade(image=lr_img)['image']
        elif hr_img is None and lr_img is not None:
            hr_img = cv2.resize(lr_img, (Config.HR_IMG_WIDTH, Config.HR_IMG_HEIGHT))
        elif lr_img is None and hr_img is None:
            lr_img = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH, 3), dtype=np.uint8)
            hr_img = np.zeros((Config.HR_IMG_HEIGHT, Config.HR_IMG_WIDTH, 3), dtype=np.uint8)

        # Apply degradations randomly during training to HR to create better LR
        if self.mode == 'train' and self.degrade and random.random() < 0.5:
            lr_img = cv2.resize(hr_img, (Config.IMG_WIDTH, Config.IMG_HEIGHT))
            lr_img = self.degrade(image=lr_img)['image']

        lr_tensor = self.transform_lr(image=lr_img)['image']
        hr_tensor = self.transform_hr(image=hr_img)['image']

        target = [Config.CHAR2IDX[c] for c in label if c in Config.CHAR2IDX]
        if len(target) == 0:
            target = [0]
            
        return lr_tensor, hr_tensor, torch.tensor(target, dtype=torch.long), len(target), label, item['track_id']

    @staticmethod
    def collate_fn(batch):
        lr_images, hr_images, targets, target_lengths, labels_text, track_ids = zip(*batch)
        lr_images = torch.stack(lr_images, 0)
        hr_images = torch.stack(hr_images, 0)
        targets = torch.cat(targets)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        return lr_images, hr_images, targets, target_lengths, labels_text, track_ids
