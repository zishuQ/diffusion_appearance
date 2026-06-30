import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_PRESETS = {
    "mot17": {
        "root": "/home/shang/datasets/MOT17/train",
        "output_dir": PROJECT_ROOT / "datasets" / "mot17_gt_fastreid",
        "config_path": "configs/MOT17/sbs_S50.yml",
        "weight_path": "weights/mot17_sbs_S50.pth",
        "classes": [1],
    },
    "mot20": {
        "root": "/home/shang/datasets/MOT20/train",
        "output_dir": PROJECT_ROOT / "datasets" / "mot20_gt_fastreid",
        "config_path": "configs/MOT20/sbs_S50.yml",
        "weight_path": "weights/mot20_sbs_S50.pth",
        "classes": [1],
    },
    "sportsmot_train": {
        "root": "/home/shang/datasets/SportsMOT/dataset/train",
        "output_dir": PROJECT_ROOT / "datasets" / "sportsmot_train_gt_fastreid",
        "config_path": "configs/SportsMOT/sbs_S50.yml",
        "weight_path": "weights/sports_sbs_S50.pth",
        "classes": [1],
    },
    "sportsmot_val": {
        "root": "/home/shang/datasets/SportsMOT/dataset/val",
        "output_dir": PROJECT_ROOT / "datasets" / "sportsmot_val_gt_fastreid",
        "config_path": "configs/SportsMOT/sbs_S50.yml",
        "weight_path": "weights/sports_sbs_S50.pth",
        "classes": [1],
    },
}


def parse_args():
    parser = argparse.ArgumentParser("Export MOT-format GT FastReID features per sequence")
    parser.add_argument("--preset", choices=sorted(DATASET_PRESETS), default=None)
    parser.add_argument("--root", default=None, help="Dataset split root containing sequence folders")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tracktrack-fastreid-dir", default="../TrackTrack/2. FastReID")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--weight-path", default=None)
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--classes", type=int, nargs="+", default=None)
    parser.add_argument("--include-unmarked", action="store_true", help="Keep rows with MOT mark != 1")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def apply_preset(args):
    preset = DATASET_PRESETS.get(args.preset or "", {})
    root = args.root or preset.get("root")
    output_dir = args.output_dir or preset.get("output_dir")
    config_path = args.config_path or preset.get("config_path")
    weight_path = args.weight_path or preset.get("weight_path")
    classes = args.classes if args.classes is not None else preset.get("classes", [1])
    if root is None or output_dir is None or config_path is None or weight_path is None:
        raise ValueError("Set --preset or provide --root, --output-dir, --config-path, and --weight-path.")
    return Path(root).expanduser().resolve(), Path(output_dir).expanduser().resolve(), config_path, weight_path, classes


def discover_sequences(root: Path):
    return sorted(p.name for p in root.iterdir() if (p / "gt" / "gt.txt").is_file() and (p / "img1").is_dir())


def load_gt_rows(gt_path: Path, classes: list[int], include_unmarked: bool):
    rows = np.loadtxt(gt_path, delimiter=",", dtype=np.float32)
    if rows.ndim == 1:
        rows = rows[None, :]
    if not include_unmarked and rows.shape[1] > 6:
        rows = rows[rows[:, 6] == 1]
    if classes and rows.shape[1] > 7:
        rows = rows[np.isin(rows[:, 7].astype(np.int32), np.array(classes, dtype=np.int32))]
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    return rows


def xywh_to_xyxy(rows):
    boxes = rows[:, 2:6].copy()
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    return boxes


def keep_valid_boxes(rows, boxes, image_shape):
    height, width = image_shape[:2]
    clipped = boxes.copy()
    clipped[:, 0::2] = np.clip(clipped[:, 0::2], 0, width)
    clipped[:, 1::2] = np.clip(clipped[:, 1::2], 0, height)
    valid = (clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])
    return rows[valid], clipped[valid], int((~valid).sum())


def export_sequence(seq: str, root: Path, output_dir: Path, embedder, overwrite: bool, classes: list[int], include_unmarked: bool):
    out_path = output_dir / f"{seq}.pkl"
    if out_path.exists() and not overwrite:
        print(f"Skip existing {out_path}")
        return

    seq_dir = root / seq
    gt_path = seq_dir / "gt" / "gt.txt"
    img_dir = seq_dir / "img1"
    rows = load_gt_rows(gt_path, classes, include_unmarked)
    records = []
    skipped_invalid = 0

    frame_ids = sorted(np.unique(rows[:, 0].astype(np.int32)).tolist())
    for frame_id in tqdm(frame_ids, desc=seq):
        frame_rows = rows[rows[:, 0].astype(np.int32) == frame_id]
        img_path = img_dir / f"{frame_id:06d}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        boxes = xywh_to_xyxy(frame_rows)
        frame_rows, boxes, skipped = keep_valid_boxes(frame_rows, boxes, img.shape)
        skipped_invalid += skipped
        if len(frame_rows) == 0:
            continue

        feats = embedder.compute_embedding(img, boxes).astype(np.float32)
        for row, box, feat in zip(frame_rows, boxes, feats):
            records.append(
                {
                    "seq": seq,
                    "frame_id": int(row[0]),
                    "track_id": int(row[1]),
                    "bbox": box.astype(np.float32),
                    "feat": feat,
                }
            )

    with out_path.open("wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {len(records)} records to {out_path}")
    if skipped_invalid:
        print(f"Skipped {skipped_invalid} invalid boxes in {seq}")


def main():
    args = parse_args()
    root, output_dir, config_path, weight_path, classes = apply_preset(args)
    fastreid_dir = Path(args.tracktrack_fastreid_dir).resolve()
    sys.path.insert(0, str(fastreid_dir))
    from fastreid.emb_computer import EmbeddingComputer

    sequences = args.sequences or discover_sequences(root)
    if not sequences:
        raise ValueError(f"No sequences found under {root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    embedder = EmbeddingComputer(
        config_path=str((fastreid_dir / config_path).resolve()),
        weight_path=str((fastreid_dir / weight_path).resolve()),
    )

    print(f"Root: {root}")
    print(f"Output: {output_dir}")
    print(f"Sequences: {len(sequences)}")
    print(f"Classes: {classes}")
    print(f"Include unmarked: {args.include_unmarked}")
    for seq in sequences:
        export_sequence(seq, root, output_dir, embedder, args.overwrite, classes, args.include_unmarked)


if __name__ == "__main__":
    main()
