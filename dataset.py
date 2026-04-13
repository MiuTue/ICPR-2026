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
            print("ERROR: Data not found.")
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
            print(f"Loading split from '{Config.VAL_SPLIT_FILE}'...")
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
        """Pairs LR and HR files within each track based on numeric suffix."""
        import re
        
        for track_path in tqdm(tracks, desc=f"Indexing {self.mode}"):
            json_path = os.path.join(track_path, "annotations.json")
            label = ""
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, list): data = data[0]
                    label = str(data.get('plate_text', data.get('license_plate', data.get('text', '')))).strip()
                except:
                    pass
            
            # Label required for training
            if self.mode == 'train' and not label:
                continue

            lr_paths = glob.glob(os.path.join(track_path, "lr-*.[jp][pn]g"))
            hr_paths = glob.glob(os.path.join(track_path, "hr-*.[jp][pn]g"))
            
            # Map by numeric suffix (e.g., 'cr-12.jpg' -> '12')
            def get_id(path):
                match = re.search(r'-(\d+)\.', os.path.basename(path))
                return match.group(1) if match else None

            hr_map = {get_id(p): p for p in hr_paths if get_id(p) is not None}
            lr_map = {get_id(p): p for p in lr_paths if get_id(p) is not None}
            
            # Combine all available IDs
            all_ids = set(hr_map.keys()) | set(lr_map.keys())
            
            for img_id in sorted(all_ids, key=lambda x: int(x) if x.isdigit() else 0):
                lr_p = lr_map.get(img_id)
                hr_p = hr_map.get(img_id)
                
                if lr_p or hr_p:
                    self.samples.append({
                        'lr_path': lr_p,
                        'hr_path': hr_p,
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

        # 1. Load HR as the ground truth
        hr_img = self._read_image(item['hr_path'])
        lr_img = self._read_image(item['lr_path'])

        # 2. Logic for generating LR
        # If training, we often prefer synthetically degraded LR for better generalization
        is_synthetic = False
        if self.mode == 'train' and self.degrade and (random.random() < 0.6 or lr_img is None):
            if hr_img is not None:
                # Proper Pipeline: HR -> Degrade -> Scale Down
                # This simulates real-world camera degradation before resolution loss
                temp_img = self.degrade(image=hr_img)['image']
                lr_img = cv2.resize(temp_img, (Config.IMG_WIDTH, Config.IMG_HEIGHT), interpolation=cv2.INTER_AREA)
                is_synthetic = True

        # Fallback Resizing if necessary
        if hr_img is None and lr_img is not None:
            hr_img = cv2.resize(lr_img, (Config.HR_IMG_WIDTH, Config.HR_IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
        elif lr_img is None and hr_img is not None:
            lr_img = cv2.resize(hr_img, (Config.IMG_WIDTH, Config.IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        elif lr_img is None and hr_img is None:
            # Absolute fallback: black images
            lr_img = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH, 3), dtype=np.uint8)
            hr_img = np.zeros((Config.HR_IMG_HEIGHT, Config.HR_IMG_WIDTH, 3), dtype=np.uint8)

        # 3. Apply final resizing and normalization via transforms
        # (Transforms handle standard augmentations and ToTensor)
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
