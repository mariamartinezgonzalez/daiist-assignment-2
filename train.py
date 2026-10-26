"""Assignment 2 training pipeline: download, finetune, evaluate, cost, save."""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

import costs
from api_key import load_api_key

WORKSPACE, PROJECT, VERSION = "iote-7emb7", "hard-hat-7785r", 1
POSITIVE = "head"
EPOCHS = int(os.environ.get("EPOCHS", 50))
FRACTION = float(os.environ.get("FRACTION", 1.0))  # <1 only for quick smoke tests
IMGSZ = int(os.environ.get("IMGSZ", 640))
BATCH = int(os.environ.get("BATCH", 16))
ROOT = Path(__file__).parent
DATA = ROOT / "data" / "hardhat"
OUT = ROOT / "outputs"
WEIGHTS = ROOT / "weights"


def download():
    if not (DATA / "data.yaml").exists():
        rf = Roboflow(api_key=load_api_key())
        rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
            "yolov11", location=str(DATA)
        )
    names = yaml.safe_load((DATA / "data.yaml").read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    cfg = {"path": str(DATA.resolve()), "train": "train/images",
           "val": "valid/images", "test": "test/images", "names": names}
    (DATA / "data_abs.yaml").write_text(yaml.safe_dump(cfg))
    return DATA / "data_abs.yaml", names


def label_stats(names):
    rows = []
    for split in ("train", "valid", "test"):
        for f in (DATA / split / "labels").glob("*.txt"):
            for line in f.read_text().splitlines():
                p = line.split()
                if len(p) >= 5:
                    rows.append((split, names[int(p[0])], float(p[3]) * float(p[4])))
    df = pd.DataFrame(rows, columns=["split", "cls", "area"])
    counts = df.groupby(["split", "cls"]).size().unstack(fill_value=0).to_dict("index")
    size = df.groupby("cls")["area"].median().to_dict()  # box area / image area
    return counts, size


def image_scores(model, pos_idx):
    rows = []
    for img in sorted((DATA / "test" / "images").glob("*")):
        lab = DATA / "test" / "labels" / (img.stem + ".txt")
        lines = lab.read_text().splitlines() if lab.exists() else []
        y = int(any(l.split() and int(l.split()[0]) == pos_idx for l in lines))
        r = model.predict(str(img), conf=0.001, imgsz=IMGSZ, verbose=False)[0]
        s = 0.0
        for c, cf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            if int(c) == pos_idx:
                s = max(s, cf)
        rows.append((img.name, y, s))
    return pd.DataFrame(rows, columns=["image", "y_true", "score"])


def cost_per_1000(tpr, fpr):
    b = costs.BASE_RATE
    return 1000 * (b * (tpr * costs.C_TP + (1 - tpr) * costs.C_FN)
                   + (1 - b) * (fpr * costs.C_FP + (1 - fpr) * costs.C_TN))


def sweep(y, s):
    P = int(y.sum())
    N = len(y) - P
    rows = []
    for t in np.linspace(0, 1, 101):
        pred = s >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        rows.append(dict(threshold=round(float(t), 2), tp=tp, fp=fp, fn=P - tp,
                         tn=N - fp, tpr=tp / P, fpr=fp / N,
                         cost_per_1000=cost_per_1000(tp / P, fp / N)))
    return pd.DataFrame(rows)


def det_metrics(model, data_yaml, names):
    b = model.val(data=str(data_yaml), split="test", imgsz=IMGSZ, batch=BATCH,
                  workers=2, verbose=False).box
    per = {names[int(c)]: dict(precision=float(b.p[i]), recall=float(b.r[i]),
                               ap50=float(b.ap50[i]), ap50_95=float(b.ap[i]))
           for i, c in enumerate(b.ap_class_index)}
    return dict(map50=float(b.map50), map50_95=float(b.map), per_class=per)


def main():
    costs.require_filled()
    OUT.mkdir(exist_ok=True)
    WEIGHTS.mkdir(exist_ok=True)
    data_yaml, names = download()
    pos_idx = names.index(POSITIVE)
    counts, size = label_stats(names)

    model = YOLO("yolo11n.pt")
    model.train(data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ, fraction=FRACTION,
                batch=BATCH, workers=2,
                project=str(ROOT / "runs"), name="hardhat", exist_ok=True, seed=0)
    shutil.copy(model.trainer.best, WEIGHTS / "best.pt")
    for f in ("confusion_matrix.png", "confusion_matrix_normalized.png", "results.png"):
        if (model.trainer.save_dir / f).exists():
            shutil.copy(model.trainer.save_dir / f, OUT / f)

    tuned = YOLO(str(WEIGHTS / "best.pt"))
    tuned_metrics = det_metrics(tuned, data_yaml, names)
    scores = image_scores(tuned, pos_idx)
    scores.to_csv(OUT / "test_scores.csv", index=False)
    sw = sweep(scores["y_true"].values, scores["score"].values)
    sw.to_csv(OUT / "cost_sweep.csv", index=False)
    best_row = sw.loc[sw["cost_per_1000"].idxmin()]

    # Baselines on the same test split.
    coco = YOLO("yolo11n.pt")  # COCO classes only
    coco_has_class = POSITIVE in coco.names.values() or "helmet" in coco.names.values()
    baselines = {
        "coco_zero_shot": {
            "has_matching_class": coco_has_class,
            "note": "COCO has no head/helmet class, so it detects none of our classes: "
                    "mAP50 = mAP50-95 = 0 by construction.",
            "map50": 0.0, "map50_95": 0.0,
            "cost_per_1000": cost_per_1000(0.0, 0.0),
        },
        "predict_nothing": {"cost_per_1000": cost_per_1000(0.0, 0.0)},
        "flag_everything": {"cost_per_1000": cost_per_1000(1.0, 1.0)},
        "human_checks_everything_per_1000": 1000 * costs.C_REVIEW,
    }

    metrics = {
        "positive_class": POSITIVE, "classes": names,
        "dataset": dict(workspace=WORKSPACE, project=PROJECT, version=VERSION),
        "instances": counts, "median_box_area_fraction": size,
        "test_images": len(scores), "test_positive_images": int(scores["y_true"].sum()),
        "finetuned": tuned_metrics, "baselines": baselines,
        "chosen_threshold": float(best_row["threshold"]),
        "min_cost_per_1000": float(best_row["cost_per_1000"]),
        "costs": {k: getattr(costs, k) for k in
                  ("C_FP", "C_FN", "C_TP", "C_TN", "C_REVIEW", "BASE_RATE")},
        "training": dict(model="yolo11n", epochs=EPOCHS, imgsz=IMGSZ, fraction=FRACTION),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()