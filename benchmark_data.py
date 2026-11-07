"""Fetch official archives and manifest the full-page development and test sets."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from py7zr import SevenZipFile

from binarization import read_gray


DUTH = "https://vc.ee.duth.gr/"
UTOPIA = "https://utopia.duth.gr/~ipratika/"
DEMOKRITOS = "https://users.iit.demokritos.gr/~bgat/"

# Group, role, official archive directory, archives, image and mask paths, ID range.
GROUPS = (
    ("dibco2011", "development", UTOPIA + "DIBCO2011/benchmark/dataset/",
     ("DIBCO11-handwritten.rar",), r"hw(\d+)\.png", r"hw(\d+)_gt\.tiff", 1, 8),
    ("dibco2013", "development", UTOPIA + "DIBCO2013/benchmark/dataset/",
     ("DIBCO2013-dataset.rar",), r"originalimages/hw(\d+)\.[^.]+", r"gtimages/hw(\d+)_estgt\.tiff", 1, 8),
    ("dibco2017", "development", DUTH + "dibco2017/benchmark/",
     ("DIBCO2017_Dataset.7z", "DIBCO2017_GT.7z"), r"dataset/(\d+)\.bmp", r"gt/(\d+)_gt\.bmp", 1, 10),
    ("hdibco2018", "development", DUTH + "h-dibco2018/benchmark/",
     ("dibco2018_Dataset.zip", "dibco2018-GT.zip"), r"dataset/(\d+)\.bmp", r"gt/(\d+)_gt\.bmp", 1, 10),
    ("dibco2009", "handwriting", DEMOKRITOS + "DIBCO2009/benchmark/",
     ("DIBC02009_Test_images-handwritten.rar", "DIBCO2009-GT-Test-images_handwritten.rar"),
     r"[^/]+/h(\d+)\.bmp", r"[^/]+/h(\d+)\.tiff", 1, 5),
    ("hdibco2010", "handwriting", DEMOKRITOS + "H-DIBCO2010/benchmark/",
     ("H_DIBCO2010_test_images.rar", "H_DIBCO2010_GT.rar"),
     r"h(\d+)\.[^.]+", r"h(\d+)_estgt\.tiff", 1, 10),
    ("hdibco2012", "handwriting", UTOPIA + "HDIBCO2012/benchmark/dataset/",
     ("H-DIBCO2012-dataset.rar",), r".*/h(\d+)\.png", r".*/h(\d+)_gt\.tif", 1, 14),
    ("hdibco2014", "handwriting", DEMOKRITOS + "HDIBCO2014/benchmark/dataset/",
     ("original_images.rar", "GT.rar"), r"h(\d+)\.png", r"h(\d+)_estgt\.tiff", 1, 10),
    ("hdibco2016", "handwriting", DUTH + "h-dibco2016/benchmark/",
     ("DIBCO2016_dataset-original.zip", "DIBCO2016_dataset-GT.zip"),
     r"dipco2016_dataset/(\d+)\.bmp", r"dipco2016_dataset_gt/(\d+)_gt\.bmp", 1, 10),
    ("dibco2019-a", "external", DUTH + "dibco2019/benchmark/",
     ("dibco2019_dataset_trackA.zip", "dibco2019_gt_trackA.zip"),
     r"dataset/(\d+)\.bmp", r"gt/(\d+)\.bmp", 1, 10),
    ("dibco2019-b", "external", DUTH + "dibco2019/benchmark/",
     ("dibco2019_dataset_trackB.zip", "dibco2019_GT_trackB.zip"),
     r"dataset/(\d+)\.bmp", r"gt/(\d+)\.bmp", 11, 20),
)


def fetch(archives, base, name):
    archive = archives / name
    if not archive.is_file():
        partial = archive.with_suffix(archive.suffix + ".part")
        with urlopen(base + name, timeout=120) as response, partial.open("wb") as target:
            shutil.copyfileobj(response, target)
        partial.replace(archive)
    return archive


def unpack(archive, destination):
    if archive.suffix == ".zip":
        with ZipFile(archive) as source:
            source.extractall(destination)
    elif archive.suffix == ".7z":
        with SevenZipFile(archive) as source:
            source.extractall(destination)
    else:
        tar = shutil.which("bsdtar") or (shutil.which("tar.exe") if os.name == "nt" else None)
        if not tar:
            raise RuntimeError("RAR extraction requires bsdtar (Linux: libarchive-tools).")
        subprocess.run([tar, "-xf", str(archive.resolve()), "-C", str(destination.resolve())], check=True)


def match_files(root, regex):
    paths = {}
    for path in root.rglob("*"):
        match = re.fullmatch(regex, path.relative_to(root).as_posix().lower())
        if path.is_file() and match:
            index = int(match[1])
            if index in paths:
                raise ValueError(f"Ambiguous {regex} index {index} in {root}")
            paths[index] = path
    return paths


def prepare(data, archives):
    archives.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        downloads = [pool.submit(fetch, archives, base, name)
                     for _, _, base, names, *_ in GROUPS for name in names]
        for future in downloads:
            future.result()
    samples = []
    duplicates = []
    seen = {}
    for group, role, base, names, image_pattern, mask_pattern, first, last in GROUPS:
        root = data / group
        root.mkdir(parents=True, exist_ok=True)
        for name in names:
            unpack(archives / name, root)
        images = match_files(root, image_pattern)
        masks = match_files(root, mask_pattern)
        kept = 0
        for i in range(first, last + 1):
            image, mask = images[i], masks[i]
            gray, label = read_gray(image), read_gray(mask)
            if gray.shape != label.shape or not np.isin(label, [0, 255]).all() or not (label == 0).any():
                raise ValueError(f"Invalid binary label: {mask}")
            digest = sha256(str(gray.shape).encode() + gray.tobytes()).hexdigest()
            key = f"{group}/{i:02d}"
            if digest in seen:
                duplicates.append({"excluded": key, "kept": seen[digest]})
                continue
            seen[digest] = key
            kept += 1
            samples.append({
                "id": key, "group": group, "role": role,
                "image": image.relative_to(data).as_posix(), "mask": mask.relative_to(data).as_posix(),
                "width": gray.shape[1], "height": gray.shape[0], "image_sha256": digest,
                "mask_sha256": sha256(label.tobytes()).hexdigest(),
            })
        print(f"{role:12} {group:12} {kept}/{last - first + 1}", flush=True)
    manifest = {"samples": samples, "exact_duplicates_excluded": duplicates}
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/benchmark"))
    parser.add_argument("--archives", type=Path, default=Path("data/archives"))
    args = parser.parse_args()
    prepare(args.data, args.archives)
