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


def _load_test_track_ids():
    """Load held-out test track IDs from Config.TEST_TRACKS_FILE."""
    path = Config.TEST_TRACKS_FILE
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f))
    return set()


class AdvancedMultiFrameDataset(Dataset):
    """
    Train / Val dataset. Test tracks (from TEST_TRACKS_FILE) are EXCLUDED.
    Splits the remaining tracks: 80% train, 20% val.
    """

    def __init__(self, root_dir=None, mode='train', split_ratio=0.8):
        self.mode = mode
        self.root_dir = root_dir or Config.DATA_ROOT
        self.samples = []
        self.test_ids = _load_test_track_ids()

        self.transform = get_train_transforms() if mode == 'train' else get_val_transforms()
        self.degrade = get_degradation_transforms() if mode == 'train' else None

        print(f"[{mode.upper()}] Scanning: {self.root_dir}")
        abs_root = os.path.abspath(self.root_dir)
        search_path = os.path.join(abs_root, "**", "track_*")
        all_tracks = sorted(glob.glob(search_path, recursive=True))

        if not all_tracks:
            print("❌ LỖI: Không tìm thấy data.")
            return

        # Exclude test tracks so they never appear in train/val
        non_test = [t for t in all_tracks if os.path.basename(t) not in self.test_ids]
        print(f"   Total tracks : {len(all_tracks)}  |  Non-test : {len(non_test)}")

        train_tracks, val_tracks = self._split_tracks(non_test, split_ratio)
        selected = train_tracks if mode == 'train' else val_tracks
        print(f"[{mode.upper()}] Selected: {len(selected)} tracks.")
        self._load_samples(selected)

    def _split_tracks(self, all_tracks, split_ratio):
        train_tracks, val_tracks = [], []

        if os.path.exists(Config.VAL_SPLIT_FILE):
            print(f"📂 Loading split from '{Config.VAL_SPLIT_FILE}'...")
            try:
                with open(Config.VAL_SPLIT_FILE, 'r') as f:
                    val_ids = set(json.load(f))
            except Exception:
                val_ids = set()
                print("⚠️ Lỗi đọc file split, sẽ tạo lại.")

            for t in all_tracks:
                if os.path.basename(t) in val_ids:
                    val_tracks.append(t)
                else:
                    train_tracks.append(t)

            if not val_tracks and len(all_tracks) > 0:
                print("⚠️ File split không khớp. Chia lại...")
                train_tracks, val_tracks = self._create_new_split(all_tracks, split_ratio)
        else:
            print("⚠️ Creating new split...")
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
        print(f"📄 Saved {len(val_ids)} val tracks to '{Config.VAL_SPLIT_FILE}'.")
        return train_tracks, val_tracks

    def _load_samples(self, tracks):
        for track_path in tqdm(tracks, desc=f"Indexing {self.mode}"):
            json_path = os.path.join(track_path, "annotations.json")
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    data = data[0]
                label = data.get('plate_text', data.get('license_plate', data.get('text', '')))
                if not label:
                    continue
                lr_files = sorted(
                    glob.glob(os.path.join(track_path, "lr-*.png")) +
                    glob.glob(os.path.join(track_path, "lr-*.jpg"))
                )
                hr_files = sorted(
                    glob.glob(os.path.join(track_path, "hr-*.png")) +
                    glob.glob(os.path.join(track_path, "hr-*.jpg"))
                )
                if len(lr_files) > 0:
                    self.samples.append({'lr_paths': lr_files, 'hr_paths': hr_files, 'label': label})
            except Exception:
                pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        label = item['label']
        use_hr = (self.mode == 'train') and (len(item['hr_paths']) > 0) and (random.random() < 0.5)
        if use_hr:
            images_list = self._load_frames(item['hr_paths'], apply_degradation=True)
        else:
            images_list = self._load_frames(item['lr_paths'], apply_degradation=False)
        images_tensor = torch.stack(images_list, dim=0)
        target = [Config.CHAR2IDX[c] for c in label if c in Config.CHAR2IDX]
        if len(target) == 0:
            target = [0]
        return images_tensor, torch.tensor(target, dtype=torch.long), len(target), label

    def _load_frames(self, paths, apply_degradation=False):
        if len(paths) < 5:
            paths = paths + [paths[-1]] * (5 - len(paths))
        else:
            paths = paths[:5]
        images_list = []
        for p in paths:
            image = cv2.imread(p)
            if image is None:
                image = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if apply_degradation and self.degrade:
                image = self.degrade(image=image)['image']
            image = self.transform(image=image)['image']
            images_list.append(image)
        return images_list

    @staticmethod
    def collate_fn(batch):
        images, targets, target_lengths, labels_text = zip(*batch)
        images = torch.stack(images, 0)
        targets = torch.cat(targets)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        return images, targets, target_lengths, labels_text


class TestDataset(Dataset):
    """
    Test dataset — reads tracks listed in Config.TEST_TRACKS_FILE.
    Those tracks live inside Config.DATA_ROOT (data/train/).
    """

    def __init__(self, root_dir=None):
        self.root_dir = root_dir or Config.DATA_ROOT
        self.transform = get_val_transforms()
        self.samples = []
        self.test_ids = _load_test_track_ids()
        self._load_samples()

    def _load_samples(self):
        print(f"[TEST] Scanning: {self.root_dir}")
        abs_root = os.path.abspath(self.root_dir)
        search_path = os.path.join(abs_root, "**", "track_*")
        all_tracks = sorted(glob.glob(search_path, recursive=True))

        # Keep ONLY tracks in TEST_TRACKS_FILE
        test_tracks = [t for t in all_tracks if os.path.basename(t) in self.test_ids]
        print(f"[TEST] Found {len(test_tracks)} / {len(all_tracks)} tracks matching test set.")

        if not test_tracks:
            print("❌ Không tìm thấy test tracks.")
            return

        for track_path in tqdm(test_tracks, desc="Indexing test"):
            json_path = os.path.join(track_path, "annotations.json")
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    data = data[0]
                label = data.get('plate_text', data.get('license_plate', data.get('text', '')))
                if not label:
                    continue
                lr_files = sorted(
                    glob.glob(os.path.join(track_path, "lr-*.png")) +
                    glob.glob(os.path.join(track_path, "lr-*.jpg"))
                )
                if len(lr_files) > 0:
                    self.samples.append({'lr_paths': lr_files, 'label': label})
            except Exception:
                pass

        print(f"[TEST] Loaded {len(self.samples)} test samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        label = item['label']
        images_list = self._load_frames(item['lr_paths'])
        images_tensor = torch.stack(images_list, dim=0)
        target = [Config.CHAR2IDX[c] for c in label if c in Config.CHAR2IDX]
        if len(target) == 0:
            target = [0]
        return images_tensor, torch.tensor(target, dtype=torch.long), len(target), label

    def _load_frames(self, paths):
        if len(paths) < 5:
            paths = paths + [paths[-1]] * (5 - len(paths))
        else:
            paths = paths[:5]
        images_list = []
        for p in paths:
            image = cv2.imread(p)
            if image is None:
                image = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transform(image=image)['image']
            images_list.append(image)
        return images_list

    @staticmethod
    def collate_fn(batch):
        images, targets, target_lengths, labels_text = zip(*batch)
        images = torch.stack(images, 0)
        targets = torch.cat(targets)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        return images, targets, target_lengths, labels_text
