import zipfile
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# Try to import tqdm for a nice progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

def extract_file(zip_path, target_dir, file_info):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract(file_info, target_dir)
        return True
    except Exception:
        return False

def fast_extract(zip_path, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.infolist()
    
    total_files = len(file_list)
    num_workers = os.cpu_count()
    print(f"Extracting {total_files} items using {num_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Create futures for all files
        futures = [executor.submit(extract_file, zip_path, target_dir, f) for f in file_list]
        
        if HAS_TQDM:
            # Use tqdm progress bar
            for _ in tqdm(as_completed(futures), total=total_files, unit="file", desc="Extracting"):
                pass
        else:
            # Fallback simple counter
            completed = 0
            for _ in as_completed(futures):
                completed += 1
                if completed % 100 == 0 or completed == total_files:
                    print(f"\rProgress: {completed}/{total_files} ({(completed/total_files)*100:.1f}%)", end="", flush=True)
            print() # New line after finishing

if __name__ == "__main__":
    zip_file = "wYe7pBJ7-train.zip"
    dest_dir = "data"
    
    if os.path.exists(zip_file):
        fast_extract(zip_file, dest_dir)
        print(f"\nDone! Data extracted to {dest_dir}")
    else:
        print(f"Error: {zip_file} not found.")
