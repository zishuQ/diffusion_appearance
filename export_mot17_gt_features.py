import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


MOT17_FRCNN_SEQUENCES = [
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
    "MOT17-11-FRCNN",
    "MOT17-13-FRCNN",
]

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser("Export MOT17 GT FastReID features per sequence")
    parser.add_argument("--mot17-root", default="/home/shang/datasets/MOT17/train")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "datasets" / "mot17_gt_fastreid"))
    parser.add_argument("--tracktrack-fastreid-dir", default="../TrackTrack/2. FastReID")
    parser.add_argument("--config-path", default="configs/MOT17/sbs_S50.yml")
    parser.add_argument("--weight-path", default="weights/mot17_sbs_S50.pth")
    parser.add_argument("--sequences", nargs="+", default=MOT17_FRCNN_SEQUENCES)
    parser.add_argument("--classes", type=int, nargs="+", default=[1])
    parser.add_argument("--include-unmarked", action="store_true", help="Keep rows with MOT mark != 1")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_gt_rows(gt_path: Path, classes: list[int], include_unmarked: bool):
    rows = np.loadtxt(gt_path, delimiter=",", dtype=np.float32)
    if rows.ndim == 1:
        rows = rows[None, :]
    if not include_unmarked:
        rows = rows[rows[:, 6] == 1]
    if classes:
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


def main():
    args = parse_args()
    fastreid_dir = Path(args.tracktrack_fastreid_dir).resolve()
    sys.path.insert(0, str(fastreid_dir))
    from fastreid.emb_computer import EmbeddingComputer

    config_path = str((fastreid_dir / args.config_path).resolve())
    weight_path = str((fastreid_dir / args.weight_path).resolve())
    embedder = EmbeddingComputer(config_path=config_path, weight_path=weight_path)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for seq in args.sequences:
        out_path = output_dir / f"{seq}.pkl"
        if out_path.exists() and not args.overwrite:
            print(f"Skip existing {out_path}")
            continue

        seq_dir = Path(args.mot17_root) / seq
        gt_path = seq_dir / "gt" / "gt.txt"
        img_dir = seq_dir / "img1"
        rows = load_gt_rows(gt_path, args.classes, args.include_unmarked)
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


if __name__ == "__main__":
    main()
