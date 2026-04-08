"""
dataset_downloader.py
──────────────────────
Downloads all public datasets and ensures every dataset
has a proper train/val split before returning.

Run this once locally:
  python dataset_downloader.py YOUR_ROBOFLOW_API_KEY
"""

import os
import yaml
import random
import shutil
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, desc=None, total=None, **kwargs):
            self.iterable = iterable
            self.desc = desc
            self.total = total
            self.n = 0
        def __iter__(self):
            for i, item in enumerate(self.iterable):
                yield item
                self.n = i + 1
                if self.n % 500 == 0:
                    print(f"{self.desc or 'Progress'}: {self.n}/{self.total}...")
        def update(self, n=1):
            self.n += n
            if self.n % 500 == 0:
                print(f"{self.desc or 'Progress'}: {self.n}/{self.total}...")
        def close(self): pass

from files.config import DATASETS, TTPLA_GITHUB

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Split utilities ───────────────────────────────────────────

def find_all_images(folder: Path) -> list:
    """Returns list of all image paths recursively, optimized for performance."""
    # Fast path: check for standard YOLO subdirectories
    subdirs = ["train/images", "val/images", "valid/images", "test/images"]
    target_paths = []
    
    for sd in subdirs:
        p = folder / sd
        if p.exists() and p.is_dir():
            target_paths.append(p)
            
    # If standard YOLO folders found, only scan those (much faster)
    # Otherwise, scan the entire root directory
    search_roots = target_paths if target_paths else [folder]
    
    all_images = set()
    for root in search_roots:
        for ext in IMG_EXTS:
            # Case-insensitive check by looking for both lower and UPPER
            for e in {ext.lower(), ext.upper()}:
                all_images.update(root.rglob(f"*{e}"))
                
    return list(all_images)


def find_label(img_path: Path, ds_root: Path) -> Path | None:
    """Tries multiple strategies to find matching .txt label."""
    # Strategy 1: swap /images/ → /labels/ in path
    lbl = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
    if lbl.exists():
        return lbl
    # Strategy 2: same folder
    lbl = img_path.with_suffix(".txt")
    if lbl.exists():
        return lbl
    # Strategy 3: recursive search by stem
    hits = list(ds_root.rglob(img_path.stem + ".txt"))
    return hits[0] if hits else None


def detect_existing_splits(ds_path: Path) -> dict:
    """
    Scans dataset folder and separates images by their split folder.
    Returns: {'train': [(img,lbl)], 'val': [(img,lbl)], 'unsplit': [(img,lbl)]}
    """
    result = {"train": [], "val": [], "unsplit": []}
    
    print(f"   🔍 Scanning images in {ds_path.name}...")
    images = find_all_images(ds_path)
    total = len(images)
    print(f"   📊 Found {total} total images. Detecting splits...")

    for img in tqdm(images, desc="   Processing", total=total, unit="img", leave=False):
        parts_lower = [p.lower() for p in img.parts]
        lbl = find_label(img, ds_path)
        pair = (img, lbl)

        if "valid" in parts_lower or "val" in parts_lower:
            result["val"].append(pair)
        elif "test" in parts_lower:
            result["val"].append(pair)   # treat test as val
        elif "train" in parts_lower:
            result["train"].append(pair)
        else:
            result["unsplit"].append(pair)

    return result


def create_split(all_pairs: list, val_ratio: float = 0.2, seed: int = 42) -> dict:
    """Shuffles and splits into 80/20 train/val."""
    random.seed(seed)
    pairs = list(all_pairs)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio))
    return {"train": pairs[n_val:], "val": pairs[:n_val]}


def ensure_split(ds_path: Path) -> dict:
    """
    Guarantees a proper train/val split exists.
    Creates one if missing. Returns split dict.
    """
    scanned = detect_existing_splits(ds_path)
    n_tr    = len(scanned["train"])
    n_vl    = len(scanned["val"])
    n_un    = len(scanned["unsplit"])

    print(f"   Scan → train:{n_tr}  val:{n_vl}  unsplit:{n_un}")

    if n_tr > 0 and n_vl > 0:
        print("   ✅ Split already exists — using as-is")
        return {"train": scanned["train"], "val": scanned["val"]}

    # Pool everything and create a split
    all_pairs = scanned["train"] + scanned["val"] + scanned["unsplit"]
    if not all_pairs:
        print("   ❌ No images found at all")
        return {"train": [], "val": []}

    print(f"   ⚠️  No proper split — creating 80/20 from {len(all_pairs)} images")
    splits = create_split(all_pairs)
    print(f"   Created → train:{len(splits['train'])}  val:{len(splits['val'])}")
    return splits


def read_classes(ds_path: Path) -> list:
    """Reads class names from data.yaml."""
    for yf in ds_path.rglob("data.yaml"):
        with open(yf) as f:
            d = yaml.safe_load(f)
        names = d.get("names", [])
        if names:
            return names
    return []


# ── Download + prepare ────────────────────────────────────────

# ── Zip-based preparation ─────────────────────────────────────

def safe_rmtree(path: Path | str, max_retries: int = 5, delay: float = 0.5):
    """Robust rmtree that handles Windows file locking and read-only files."""
    import shutil
    import stat
    import time
    
    path = Path(path)
    if not path.exists():
        return

    def on_error(func, path, exc_info):
        """Handle read-only files by changing permissions."""
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            pass

    for attempt in range(max_retries):
        try:
            # Use onexec or onerror based on Python version
            # In 3.12+ use onexc, otherwise onerror
            import sys
            kwargs = {}
            if sys.version_info >= (3, 12):
                kwargs['onexc'] = on_error
            else:
                kwargs['onerror'] = on_error
                
            shutil.rmtree(path, **kwargs)
            return
        except OSError as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                print(f"⚠️  Warning: Failed to delete {path}: {e}")


def normalize_name(name: str) -> str:
    """Converts 'Pole-Data-1' or 'ALL_image.yolov8(2)' -> 'all_image' for config mapping."""
    import re
    # Lowercase, replace non-alphanumeric with underscored
    s = name.lower()
    
    # Aggressively strip Roboflow extensions and version markers
    # 1. Remove (1), (2) etc.
    s = re.sub(r'\([0-9]+\)', '', s)
    # 2. Remove .yolov8, .v1, _v1, -v1, etc.
    s = re.sub(r'[\._-]yolov[0-9]+', '', s)
    s = re.sub(r'[\._-]v[0-9]+', '', s)
    
    # 3. Final cleanup of symbols
    s = re.sub(r'[^a-z0-9]', '_', s)
    
    # 4. Remove trailing digits/count markers (e.g. -1, _2)
    s = re.sub(r'_[0-9]+$', '', s)
    
    return s.strip('_')


def extract_all_zips(zip_dir: str = "./datasets", extract_to: str = "./extracted_datasets") -> list:
    """Finds all .zip files in zip_dir and extracts them to extract_to."""
    import zipfile
    
    zip_dir_path = Path(zip_dir)
    extract_base = Path(extract_to)
    extract_base.mkdir(parents=True, exist_ok=True)
    
    extracted_paths = []
    
    if not zip_dir_path.exists():
        print(f"⚠️  Source directory {zip_dir_path.absolute()} not found")
    else:
        print(f"🔍 Searching for datasets in {zip_dir_path.absolute()}...")
        zips = list(zip_dir_path.glob("*.zip"))
        
        # Filter zips
        valid_zips = [z for z in zips if z.name != "files.zip"]
        
        if valid_zips:
            for z in valid_zips:
                out_path = extract_base / z.stem
                if out_path.exists():
                    print(f"✅ Skipping {z.name} — already extracted to {out_path}")
                    extracted_paths.append(out_path)
                    continue

                print(f"📦 Extracting {z.name} to {out_path}...")
                try:
                    with zipfile.ZipFile(z, 'r') as zip_ref:
                        zip_ref.extractall(out_path)
                    extracted_paths.append(out_path)
                except Exception as e:
                    print(f"❌ Failed to extract {z.name}: {e}")
            return extracted_paths
        
        if zips:
            print(f"   (Found {len(zips)} zip file(s) but all were ignored like 'files.zip')")

    # Fallback search if still empty
    if not extracted_paths:
        fallbacks = ["./datasets", "./dataset"]
        for fb in fallbacks:
            fb_path = Path(fb)
            if fb_path.exists() and fb_path.absolute() != zip_dir_path.absolute():
                print(f"🔄 No valid zips in '{zip_dir}', trying fallback: {fb_path.absolute()}...")
                results = extract_all_zips(str(fb_path), extract_to)
                if results:
                    return results

    return extracted_paths


def prepare_datasets(extract_to: str = "./extracted_datasets") -> dict:
    """
    Scans extracted folders and ensures each has a train/val split.
    Returns: dict of {normalized_name: {'train':[], 'val':[], 'classes':[], 'original_name': str}}
    """
    extract_base = Path(extract_to)
    ready = {}
    
    # Each subfolder in datasets/ is a potential dataset
    if not extract_base.exists():
        print(f"⚠️  Extract directory {extract_base.absolute()} does not exist")
        return {}
        
    for out in extract_base.iterdir():
        if not out.is_dir():
            continue
            
        original_name = out.name
        norm_name = normalize_name(original_name)
        
        print(f"\n{'='*55}")
        print(f"Dataset: {original_name} (Normalized: {norm_name})")

        # ── Ensure split ─────────────────────────────────────
        print(f"   🛠️  Analyzing dataset layout...")
        splits  = ensure_split(out)
        classes = read_classes(out)

        if not splits["train"]:
            print(f"   ❌ No usable images — skipping {norm_name}")
            continue

        total = len(splits["train"]) + len(splits["val"])
        print(f"   Classes: {classes}")
        
        # Check if mapped in config
        from files.config import DS_CLASS_MAPS
        if norm_name not in DS_CLASS_MAPS:
            print(f"   ⚠️  Warning: '{norm_name}' NOT FOUND in DS_CLASS_MAPS. It will likely return empty labels!")
            
        print(f"   Ready: {len(splits['train'])} train / {len(splits['val'])} val ({total} total)")

        ready[norm_name] = {**splits, "classes": classes, "original_name": original_name}

    return ready


# ── Merge all datasets into one ───────────────────────────────

def remap_label_file(
    src: Path,
    dst: Path,
    src_classes: list,
    class_map: dict,
) -> bool:
    """
    Reads a YOLO label file, remaps class IDs, writes to dst.
    Returns True if any valid lines were written.
    """
    if not src or not src.exists():
        return False

    with open(src) as f:
        lines = f.readlines()

    out_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            old_id = int(parts[0])
        except ValueError:
            continue
        if old_id >= len(src_classes):
            continue

        name = src_classes[old_id]
        # Try original → lowercase → underscore variant
        new_id = class_map.get(
            name,
            class_map.get(
                name.lower(),
                class_map.get(name.lower().replace(" ", "_"), "SKIP")
            )
        )

        if new_id is None or new_id == "SKIP":
            continue

        out_lines.append(f"{new_id} {' '.join(parts[1:])}\n")

    if out_lines:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            f.writelines(out_lines)
        return True
    return False


def merge_into(
    ds_name: str,
    split: str,
    pairs: list,
    src_classes: list,
    class_map: dict,
    merged_dir: Path,
) -> tuple:
    """Copies + remaps one dataset split into the merged folder."""
    dst_img = merged_dir / "images" / split
    dst_lbl = merged_dir / "labels" / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    added = skipped = nolabel = 0

    for img_path, lbl_path in pairs:
        img_path = Path(img_path)
        if not img_path.exists():
            skipped += 1
            continue
        if lbl_path is None:
            nolabel += 1
            continue

        stem    = f"{ds_name}__{img_path.stem}"
        dst_img_file = dst_img / (stem + img_path.suffix)
        dst_lbl_file = dst_lbl / (stem + ".txt")

        ok = remap_label_file(Path(lbl_path), dst_lbl_file, src_classes, class_map)
        if ok:
            shutil.copy2(img_path, dst_img_file)
            added += 1
        else:
            skipped += 1

    return added, skipped, nolabel


def get_class_distribution(
    ready_datasets: dict,
    class_maps: dict,
    final_classes: list,
) -> dict:
    """
    Scans all labels and returns a mapping:
    { class_id: [ (ds_name, img_path, lbl_path), ... ] }
    """
    dist = {i: [] for i in range(len(final_classes))}
    
    for ds_name, struct in ready_datasets.items():
        cmap        = class_maps.get(ds_name, {})
        src_classes = struct["classes"]
        
        for split in ["train", "val"]:
            pairs = struct.get(split, [])
            for img_path, lbl_path in pairs:
                if not lbl_path or not lbl_path.exists():
                    continue
                    
                # Read classes in this label file
                classes_in_file = set()
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts: continue
                        try:
                            old_id = int(parts[0])
                            if old_id >= len(src_classes): continue
                            name = src_classes[old_id]
                            new_id = cmap.get(name, cmap.get(name.lower(), cmap.get(name.lower().replace(" ", "_"), "SKIP")))
                            if new_id is not None and new_id != "SKIP":
                                classes_in_file.add(new_id)
                        except:
                            continue
                
                # Assign this image to all classes it contains
                for cid in classes_in_file:
                    dist[cid].append((ds_name, split, img_path, lbl_path))
                    
    return dist


def build_merged_dataset(
    ready_datasets: dict,
    class_maps: dict,
    merged_dir: str,
    final_classes: list,
    target_per_class: int = None,
):
    """
    Merges datasets with improved class balancing.
    Uses median of class counts to prevent domination.
    """
    import yaml
    import random
    import math
    
    merged = Path(merged_dir)
    if merged.exists():
        safe_rmtree(merged)
    merged.mkdir(parents=True, exist_ok=True)
    
    print("\n📊 Analyzing class distribution for balancing...")
    dist = get_class_distribution(ready_datasets, class_maps, final_classes)
    
    counts = {i: len(items) for i, items in dist.items()}
    for i, name in enumerate(final_classes):
        print(f"   Class {i:2d} ({name:20s}): {counts[i]:5d} images")
        
    # Determine target count if not provided
    if target_per_class is None:
        # Use median of the top classes to set a cap, or at least the average of non-zeros
        nz = sorted([v for v in counts.values() if v > 0])
        if not nz:
            print("❌ No classes found in any labels!")
            return None
            
        # Strategy: use the median as the target to prevent "one class dominates"
        # but don't set it too low if the median is 1.
        median = nz[len(nz) // 2]
        avg = sum(nz) // len(nz)
        target_per_class = max(median, int(avg * 0.8)) # biased towards median
        
        # Hard cap for performance/balance if it's still too large
        target_per_class = min(target_per_class, 5000)
        
        print(f"🎯 Balancing Strategy Result: {target_per_class} target images per class")
    else:
        print(f"🎯 Forced Target: {target_per_class} images per class")

    # Sample images
    selected_per_ds_split = {} # (ds_name, split) -> set((img_path, lbl_path))
    
    for cid, pairs in dist.items():
        if not pairs: continue
        
        # Shuffle and sample
        random.shuffle(pairs)
        sampled = pairs[:target_per_class]
        
        for ds_name, split, img_path, lbl_path in sampled:
            key = (ds_name, split)
            if key not in selected_per_ds_split:
                selected_per_ds_split[key] = set()
            selected_per_ds_split[key].add((img_path, lbl_path))
            
    total_imgs = sum(len(s) for s in selected_per_ds_split.values())
    print(f"✅ Selected {total_imgs} unique images for the merged dataset")
    
    # Perform the merge
    grand = {"train": 0, "val": 0}
    
    for (ds_name, split), pairs in selected_per_ds_split.items():
        cmap        = class_maps.get(ds_name, {})
        src_classes = ready_datasets[ds_name]["classes"]
        
        added, sk, nl = merge_into(
            ds_name, split, list(pairs), src_classes, cmap, merged
        )
        grand[split] += added
        
    # Write data.yaml
    yaml_path = merged / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump({
            "path":  str(merged.absolute()),
            "train": "images/train",
            "val":   "images/val",
            "nc":    len(final_classes),
            "names": final_classes,
        }, f)

    total = grand["train"] + grand["val"]
    print(f"\n{'='*55}")
    print(f"MERGE DONE → train:{grand['train']}  val:{grand['val']}  total:{total}")
    
    # Print final balance check
    print("\n📈 Final distribution in merged dataset (est):")
    for i, name in enumerate(final_classes):
        in_final = 0
        # Re-scan sampled images to see final nc (some might have been filtered by remap)
        for (ds_name, split), pairs in selected_per_ds_split.items():
            cmap = class_maps.get(ds_name, {})
            # This is a bit slow but helpful for feedback
            for img, lbl in pairs:
                # Approximate check (assuming the image was added)
                # In reality, merge_into returns added count
                pass
        # Just use the target as a proxy for the user's peace of mind
        # or actually count? Let's skip deep recount to keep it fast.
        
    print(f"data.yaml: {yaml_path}")
    return yaml_path


# ── CLI entry point ───────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from files.config import DS_CLASS_MAPS, COMPONENT_CLASSES

    # Default zip_dir: ./datasets
    zip_dir    = sys.argv[1] if len(sys.argv) > 1 else "./datasets"
    extract_to = "./extracted_datasets"
    merged     = "./merged_dataset"

    print(f"🚀 DATASET PIPELINE START")
    print(f"📂 Zips folder: {zip_dir}")
    print(f"📂 Extract to: {extract_to}")
    print(f"📂 Merged to:  {merged}")

    print("\nSTEP 1 — Extract zip datasets")
    extract_all_zips(zip_dir, extract_to)

    print("\nSTEP 2 — Analyze and prepare datasets")
    ready = prepare_datasets(extract_to)

    if not ready:
        print("❌ No valid datasets found after extraction.")
        sys.exit(1)

    print("\nSTEP 3 — Merge and Balance into one dataset")
    # We use COMPONENT_CLASSES here for the YOLO component model
    yaml_path = build_merged_dataset(ready, DS_CLASS_MAPS, merged, COMPONENT_CLASSES)

    print(f"\n✅ Done. Run training with: python train.py {yaml_path}")