"""Select parameters on development data, then score the frozen methods on held-out scans."""

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import cv2
import numpy as np

from binarization import (
    adaptive,
    correct_background,
    median_otsu,
    otsu,
    pixel_metrics,
    read_gray,
    sauvola,
    suppress_border_strips,
)


E = "E_corrected_adaptive"
F = "F_adaptive_edge"
G = "G_corrected_sauvola"
BASELINES = ("A_otsu", "B_adaptive", "C_corrected_otsu", "D_corrected_median_otsu")
METHODS = (*BASELINES, E, F, G)
COMPARISONS = (("C", "A"), ("D", "C"), ("E", "C"), ("F", "E"), ("G", "C"))
WIDTHS = (0.02, 0.05, 0.09)


def save_png(path, image):
    encoded, data = cv2.imencode(".png", image)
    if not encoded:
        raise ValueError(f"Cannot encode {path}")
    data.tofile(path)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_samples(data, samples):
    for sample in samples:
        gray = read_gray(data / sample["image"])
        label = read_gray(data / sample["mask"])
        image_hash = sha256(str(gray.shape).encode() + gray.tobytes()).hexdigest()
        if image_hash != sample["image_sha256"] or sha256(label.tobytes()).hexdigest() != sample["mask_sha256"]:
            raise ValueError(f"Data changed after preparation: {sample['id']}")
        yield sample, gray, label == 0


def stage_one(data, samples, output):
    development = [sample for sample in samples if sample["role"] == "development"]
    local = {params: [] for params in product(WIDTHS, (2, 5, 10, 15))}
    sauvola_scores = {params: [] for params in product(WIDTHS, (0.08, 0.15, 0.25))}
    # The original four baselines retain the parameters from the first experiment.
    for _, gray, truth in read_samples(data, development):
        corrected, _ = correct_background(gray, 0.025)
        for params, scores in local.items():
            scores.append(pixel_metrics(adaptive(corrected, *params), truth)["f1"])
        for params, scores in sauvola_scores.items():
            scores.append(pixel_metrics(sauvola(corrected, *params), truth)["f1"])

    def rank(candidates):
        return sorted(
            ({"params": params, "mean_f1": mean(scores)} for params, scores in candidates.items()),
            key=lambda item: -item["mean_f1"],
        )

    local_ranked, sauvola_ranked = rank(local), rank(sauvola_scores)
    edge = {params: [] for params in product((0.03, 0.06, 0.09), (0.30, 0.18))}
    for _, gray, truth in read_samples(data, development):
        corrected, _ = correct_background(gray, 0.025)
        mask = adaptive(corrected, *local_ranked[0]["params"])
        for params, scores in edge.items():
            scores.append(pixel_metrics(suppress_border_strips(mask, *params), truth)["f1"])
    edge_ranked = rank(edge)
    lock = {
        "development_ids": [sample["id"] for sample in development],
        "test_ids": [sample["id"] for sample in samples if sample["role"] != "development"],
        "baseline_parameters": {"B": [0.05, 11], "background_fraction": 0.025, "median_width": 3},
        "E": local_ranked[0], "F": edge_ranked[0], "G": sauvola_ranked[0],
        "search": {"E": local_ranked, "F": edge_ranked, "G": sauvola_ranked},
    }
    (output / "frozen_settings.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    for method in ("E", "F", "G"):
        print(f"{method}: {lock[method]}", flush=True)


def predictions(gray, lock):
    baseline = lock["baseline_parameters"]
    corrected, _ = correct_background(gray, baseline["background_fraction"])
    local = adaptive(corrected, *lock["E"]["params"])
    return dict(zip(METHODS, (
        otsu(gray), adaptive(gray, *baseline["B"]), otsu(corrected), median_otsu(corrected),
        local, suppress_border_strips(local, *lock["F"]["params"]), sauvola(corrected, *lock["G"]["params"]),
    )))


def edge_band(shape):
    height, width = shape
    y = max(1, round(0.08 * height))
    x = max(1, round(0.08 * width))
    mask = np.ones(shape, dtype=bool)
    mask[y:height - y, x:width - x] = False
    return mask


def summarize(rows):
    summary = []
    groups = ("all", "handwriting", "external", *dict.fromkeys(row["group"] for row in rows))
    for group in groups:
        for method in METHODS:
            selected = [row for row in rows if row["method"] == method
                        and (group == "all" or group in (row["role"], row["group"]))]
            summary.append({
                "group": group, "method": method, "n": len(selected),
                **{key: mean(row[key] for row in selected) for key in ("precision", "recall", "f1")},
                "edge_fp": sum(row["edge_fp"] for row in selected),
                "edge_fn": sum(row["edge_fn"] for row in selected),
            })
    return summary


def paired_comparisons(rows):
    scores = {row["id"]: {} for row in rows}
    groups = defaultdict(list)
    for row in rows:
        scores[row["id"]][row["method"][0]] = row["f1"]
        if row["method"] == BASELINES[0]:
            groups[row["role"], row["group"]].append(row["id"])
    rng = np.random.default_rng(24)
    results = []
    for role in ("all", "handwriting", "external"):
        cohorts = [ids for (cohort_role, _), ids in groups.items() if role in ("all", cohort_role)]
        for new, old in COMPARISONS:
            deltas = [np.array([scores[i][new] - scores[i][old] for i in ids]) for ids in cohorts]
            paired = np.concatenate(deltas)
            # Resample entire pages within each fixed source group, pairing methods.
            resampled = sum(rng.choice(group, size=(5000, len(group)), replace=True).sum(axis=1)
                            for group in deltas) / len(paired)
            low, high = np.quantile(resampled, [0.025, 0.975])
            results.append({
                "group": role, "comparison": f"{new}-{old}", "n": len(paired),
                "mean_delta_f1": float(paired.mean()), "ci95_low": float(low), "ci95_high": float(high),
                "wins": int((paired > 1e-12).sum()), "losses": int((paired < -1e-12).sum()),
                "ties": int((np.abs(paired) <= 1e-12).sum()),
            })
    return results


def stage_two(data, samples, output, save_images):
    lock = json.loads((output / "frozen_settings.json").read_text(encoding="utf-8"))
    test = [sample for sample in samples if sample["role"] != "development"]
    development_ids = [sample["id"] for sample in samples if sample["role"] == "development"]
    if lock["test_ids"] != [sample["id"] for sample in test] or lock["development_ids"] != development_ids:
        raise ValueError("Frozen dataset split differs from manifest.")
    rows = []
    for sample, gray, truth in read_samples(data, test):
        edge = edge_band(gray.shape)
        for method, mask in predictions(gray, lock).items():
            row = {key: sample[key] for key in ("id", "group", "role")}
            rows.append({
                **row, "method": method, **pixel_metrics(mask, truth),
                "edge_fp": int(np.count_nonzero(mask & ~truth & edge)),
                "edge_fn": int(np.count_nonzero(~mask & truth & edge)),
            })
            if save_images:
                destination = output / "images" / sample["id"]
                destination.mkdir(parents=True, exist_ok=True)
                save_png(destination / f"{method}.png", (~mask).astype(np.uint8) * 255)
    summary = summarize(rows)
    write_csv(output / "per_image.csv", rows)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "paired.csv", paired_comparisons(rows))
    for row in summary[:21]:
        print(f"{row['group']:12} {row['method']:25} n={row['n']:2} F1={row['f1']:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("tune", "test"))
    parser.add_argument("--data", type=Path, default=Path("data/benchmark"))
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    cv2.setNumThreads(2)
    start = perf_counter()
    if args.phase == "tune":
        stage_one(args.data, manifest["samples"], args.output)
    else:
        stage_two(args.data, manifest["samples"], args.output, args.save_images)
    print(f"{args.phase}: {perf_counter() - start:.1f} s", flush=True)
