"""Train mask-proposal classifiers and evaluate them against YOLO-MRC."""
import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import cv2
import joblib
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
HOG = cv2.HOGDescriptor((32, 32), (16, 16), (8, 8), (8, 8), 9)


def resolve(base, value):
    return (base / value).resolve()


def split_paths(base, settings):
    images = resolve(base, settings["images"])
    labels = resolve(base, settings["labels"])
    if not images.is_dir() or not labels.is_dir():
        raise ValueError(f"Missing image or label directory: {images}, {labels}")
    if settings.get("manifest"):
        manifest = resolve(base, settings["manifest"])
        paths = [resolve(images, line.strip()) for line in manifest.read_text(
            encoding="utf-8-sig").splitlines() if line.strip()]
    else:
        paths = sorted(p.resolve() for p in images.rglob("*") if p.suffix.lower() in EXTS)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError(f"Empty split or duplicate image paths: {images}")
    return [(p, labels / p.relative_to(images).with_suffix(".txt")) for p in paths]


def open_sample(sample):
    path, label = sample
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    height, width = image.shape[:2]
    boxes = []
    for line in label.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        cls, x, y, w, h = map(float, line.split())
        if cls not in (0, 1) or not all(0 <= v <= 1 for v in (x, y, w, h)) or min(w, h) <= 0:
            raise ValueError(f"Invalid YOLO label: {label}: {line}")
        box = np.array([(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height])
        boxes.append((int(cls), np.clip(box, 0, [width, height, width, height])))
    return image, boxes


def iou(a, b):
    overlap = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    union = np.prod(a[2:]-a[:2]) + np.prod(b[2:]-b[:2]) - np.prod(overlap)
    return float(np.prod(overlap) / union) if union > 0 else 0.0


def mask_box(mask):
    y, x = np.where(mask)
    if not len(x):
        return None
    return np.array([x.min(), y.min(), x.max()+1, y.max()+1])


def make_segmenter(spec, base, device):
    if spec["kind"] == "cellpose":
        import torch
        from cellpose.models import CellposeModel

        weights = spec.get("checkpoint")
        model = CellposeModel(device=torch.device(device),
                              pretrained_model=str(resolve(base, weights)) if weights else "cpsam")

        def segment(rgb):
            mask = np.asarray(model.eval(rgb, channel_axis=-1)[0])
            if mask.shape != rgb.shape[:2]:
                raise ValueError(f"Unexpected Cellpose mask shape: {mask.shape}")
            for label in np.unique(mask):
                if label > 0:
                    yield mask == label
        return segment

    if spec["kind"] == "sam":
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        model = sam_model_registry[spec.get("model_type", "vit_b")](
            checkpoint=str(resolve(base, spec["checkpoint"])))
        model.to(device).eval()
        generator = SamAutomaticMaskGenerator(model, **spec.get("generator", {}))
        return lambda rgb: (item["segmentation"] for item in generator.generate(rgb))

    raise ValueError(f"Unknown segmentation backend: {spec['kind']}")


def features(image, box):
    x1, y1, x2, y2 = box.astype(int)
    patch = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(cv2.resize(patch, (32, 32)), cv2.COLOR_BGR2GRAY)
    return np.r_[HOG.compute(gray).ravel(), gray.mean()/255, gray.std()/255,
                 (x2-x1)/image.shape[1], (y2-y1)/image.shape[0]]


def boxes_for(image, segment, min_pixels):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for mask in segment(rgb):
        if np.count_nonzero(mask) >= min_pixels:
            yield mask_box(mask)


def train_classifier(samples, segment, min_pixels):
    rows, targets = [], []
    for sample in samples:
        image, truth = open_sample(sample)
        for box in boxes_for(image, segment, min_pixels):
            best = max(((iou(box, gt), cls) for cls, gt in truth), default=(0.0, 2))
            if best[0] >= 0.5 or best[0] < 0.3:
                rows.append(features(image, box))
                targets.append(best[1] if best[0] >= 0.5 else 2)
    if set(targets) != {0, 1, 2}:
        raise ValueError(f"Training proposals must contain both foreground classes and background. Found {set(targets)}.")
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)
    classifier.fit(rows, targets)
    return classifier, np.bincount(targets, minlength=3).tolist()


def suppress(predictions, threshold=0.7, limit=100):
    kept = []
    for cls, score, box in sorted(predictions, key=lambda p: -p[1]):
        if any(cls == other and iou(box, old) > threshold for other, _, old in kept):
            continue
        kept.append((cls, score, box))
        if len(kept) == limit:
            break
    return kept


def predict_masks(image, segment, min_pixels, classifier):
    boxes = list(boxes_for(image, segment, min_pixels))
    if not boxes:
        return []
    scores = classifier.predict_proba([features(image, box) for box in boxes])
    return suppress([(cls, float(prob[cls]), box)
                     for prob, box in zip(scores, boxes) for cls in (0, 1)])


def make_detector(spec, base, device):
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))
    from ultralytics import RTDETR, YOLO

    model_cls = {"rtdetr": RTDETR, "yolo": YOLO}[spec.get("kind", "rtdetr")]
    model = model_cls(str(resolve(base, spec["checkpoint"])))

    def detect(image):
        result = model.predict(image, imgsz=spec.get("imgsz", 640), device=device,
                               conf=0.001, max_det=100, verbose=False)[0]
        boxes = result.boxes.cpu().numpy()
        if not set(boxes.cls.astype(int)).issubset({0, 1}):
            raise ValueError("Detector has unexpected class IDs")
        return [(int(c), float(s), b) for c, s, b in zip(boxes.cls, boxes.conf, boxes.xyxy)]
    return detect


def average_precision(truth, predictions, cls, overlap=0.5):
    gt = [[box for label, box in items if label == cls] for items in truth]
    count = sum(map(len, gt))
    if count == 0:
        return None
    ranked = sorted(((score, index, box) for index, items in enumerate(predictions)
                     for label, score, box in items if label == cls), key=lambda item: -item[0])
    matched = [set() for _ in gt]
    hits = []
    for _, index, box in ranked:
        best = max(((iou(box, candidate), j) for j, candidate in enumerate(gt[index])
                    if j not in matched[index]), default=(0.0, -1))
        hit = best[0] >= overlap
        if hit:
            matched[index].add(best[1])
        hits.append(hit)
    if not hits:
        return 0.0
    tp = np.cumsum(hits)
    recall = tp / count
    precision = tp / np.arange(1, len(tp)+1)
    return float(np.mean([precision[recall >= r].max(initial=0) for r in np.linspace(0, 1, 101)]))


def evaluate(samples, predict, names, destination):
    truth, predictions, rows = [], [], []
    for path, label in samples:
        image, targets = open_sample((path, label))
        detections = predict(image)
        truth.append(targets)
        predictions.append(detections)
        rows.append({"image": str(path), "detections": [
            {"class_id": cls, "confidence": score, "xyxy": box.tolist()}
            for cls, score, box in detections]})
    ap = [average_precision(truth, predictions, cls) for cls in range(len(names))]
    (destination / "predictions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return {"test_images": len(samples), "AP50": dict(zip(names, ap)),
            "mAP50": float(np.mean(ap)) if None not in ap else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    base = config_path.parent
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    train = split_paths(base, cfg["train"])
    test = split_paths(base, cfg["test"])
    train_hashes = {hashlib.sha256(path.read_bytes()).hexdigest() for path, _ in train}
    if any(hashlib.sha256(path.read_bytes()).hexdigest() in train_hashes for path, _ in test):
        raise ValueError("Identical image content is present in both training and test sets.")
    names = cfg["classes"]
    if len(names) != 2 or len(set(names)) != 2:
        raise ValueError("Provide two distinct class names in YOLO ID order.")
    # Validate all inputs before running expensive proposal generation.
    for sample in train + test:
        open_sample(sample)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.yaml").write_bytes(config_path.read_bytes())
    (args.out / "test-images.txt").write_text("\n".join(str(p) for p, _ in test)+"\n", encoding="utf-8")
    device = cfg.get("device", "cpu")
    report = {}
    for name, spec in cfg["segmenters"].items():
        directory = args.out / name
        directory.mkdir()
        segment = make_segmenter(spec, base, device)
        min_pixels = cfg.get("min_mask_pixels", 4)
        classifier, counts = train_classifier(train, segment, min_pixels)
        joblib.dump(classifier, directory / "classifier.joblib")
        report[name] = evaluate(test, lambda im: predict_masks(im, segment, min_pixels, classifier), names, directory)
        report[name]["train_proposals"] = dict(zip([*names, "background"], counts))
        del segment
    directory = args.out / "yolo_mrc"
    directory.mkdir()
    report["yolo_mrc"] = evaluate(test, make_detector(cfg["detector"], base, device), names, directory)
    report["packages"] = {package: version(package) for package in
                          ("cellpose", "torch", "numpy", "scikit-learn")}
    output = json.dumps(report, indent=2, ensure_ascii=False)
    (args.out / "metrics.json").write_text(output+"\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
