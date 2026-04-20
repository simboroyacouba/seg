"""
Évaluation complète du modèle Keras/TensorFlow
Métriques IDENTIQUES à Mask R-CNN, DeepLabV3+ et YOLO26-seg pour comparaison équitable:
- mAP@50 (IoU threshold = 0.5)
- mAP@50:95 (IoU thresholds de 0.5 à 0.95)
- Precision, Recall, F1-Score
- IoU moyen
"""

import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask_utils
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
import warnings
import yaml
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION (identique à Mask R-CNN, DeepLabV3+ et YOLO26-seg)
# =============================================================================

def load_classes(yaml_path=None):
    path = yaml_path or os.getenv("CLASSES_FILE", "classes.yaml")
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data['classes']

CONFIG = {
    "images_dir": os.getenv("SEGMENTATION_DATASET_IMAGES_DIR"),
    "annotations_file": os.getenv("SEGMENTATION_DATASET_ANNOTATIONS_FILE"),
    "classes_file": os.getenv("CLASSES_FILE", "classes.yaml"),
    "model_path": os.getenv("SEGMENTATION_MODEL_PATH", "./output/best_model.keras"),
    "output_dir": "./evaluation",

    "classes": load_classes(),

    "score_threshold": 0.5,
    "iou_thresholds": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    "image_size": 224,
}


# =============================================================================
# CALCUL DES MÉTRIQUES (identique à Mask R-CNN et DeepLabV3+)
# =============================================================================

def calculate_iou_masks(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def calculate_iou_boxes(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def calculate_ap(recalls, precisions):
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[0.0], precisions, [0.0]])
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    return float(ap)


class MetricsCalculator:
    def __init__(self, num_classes, class_names, iou_thresholds):
        self.num_classes = num_classes
        self.class_names = class_names
        self.iou_thresholds = iou_thresholds
        self.reset()

    def reset(self):
        self.detections = defaultdict(list)
        self.n_gts = defaultdict(int)
        self._img_idx = 0
        self.box_ious = []
        self.mask_ious = []

    def add_image(self, pred_boxes, pred_labels, pred_masks,
                  gt_boxes, gt_labels, gt_masks):
        for class_id in range(1, self.num_classes):
            pred_cls = pred_labels == class_id
            gt_cls = gt_labels == class_id

            pred_b = pred_boxes[pred_cls]
            pred_m = [pred_masks[i] for i, v in enumerate(pred_cls) if v]
            gt_b = gt_boxes[gt_cls]
            gt_m = [gt_masks[i] for i, v in enumerate(gt_cls) if v]

            n_pred = len(pred_b)
            n_gt = len(gt_b)
            self.n_gts[class_id] += n_gt

            if n_pred == 0:
                continue

            iou_matrix = np.zeros((n_pred, n_gt))
            for i in range(n_pred):
                for j in range(n_gt):
                    box_iou_val = calculate_iou_boxes(pred_b[i], gt_b[j]) if len(pred_b) and len(gt_b) else 0
                    if pred_m and gt_m:
                        mask_iou_val = calculate_iou_masks(pred_m[i], gt_m[j])
                        iou_matrix[i, j] = (box_iou_val + mask_iou_val) / 2
                        self.mask_ious.append(mask_iou_val)
                    else:
                        iou_matrix[i, j] = box_iou_val
                    self.box_ious.append(box_iou_val)

            # Segmentation sémantique : score = 1.0 (pas de score de confiance)
            for i in range(n_pred):
                self.detections[class_id].append({
                    'score': 1.0,
                    'ious': iou_matrix[i].copy(),
                    'img_idx': self._img_idx
                })

        self._img_idx += 1

    def _compute_ap(self, class_id, iou_thresh):
        n_gt = self.n_gts[class_id]
        dets = self.detections[class_id]
        if n_gt == 0 or not dets:
            return 0.0

        dets_sorted = sorted(dets, key=lambda d: d['score'], reverse=True)
        matched = defaultdict(set)
        tp_list, fp_list = [], []

        for d in dets_sorted:
            ious = d['ious']
            img_idx = d['img_idx']
            best_iou, best_j = 0.0, -1

            for j, v in enumerate(ious):
                if j not in matched[img_idx] and v > best_iou:
                    best_iou, best_j = v, j

            if best_iou >= iou_thresh:
                tp_list.append(1); fp_list.append(0)
                matched[img_idx].add(best_j)
            else:
                tp_list.append(0); fp_list.append(1)

        tp_cum = np.cumsum(tp_list, dtype=float)
        fp_cum = np.cumsum(fp_list, dtype=float)
        recalls = tp_cum / n_gt
        precisions = tp_cum / (tp_cum + fp_cum)
        return calculate_ap(recalls, precisions)

    def _compute_prf(self, class_id, iou_thresh):
        n_gt = self.n_gts[class_id]
        dets = self.detections[class_id]
        n_pred = len(dets)

        if n_gt == 0 and n_pred == 0:
            return {'TP': 0, 'FP': 0, 'FN': 0, 'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}
        if n_gt == 0:
            return {'TP': 0, 'FP': n_pred, 'FN': 0, 'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}
        if n_pred == 0:
            return {'TP': 0, 'FP': 0, 'FN': n_gt, 'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}

        dets_sorted = sorted(dets, key=lambda d: d['score'], reverse=True)
        matched = defaultdict(set)
        tp = fp = 0

        for d in dets_sorted:
            ious = d['ious']
            img_idx = d['img_idx']
            best_iou, best_j = 0.0, -1

            for j, v in enumerate(ious):
                if j not in matched[img_idx] and v > best_iou:
                    best_iou, best_j = v, j

            if best_iou >= iou_thresh:
                tp += 1; matched[img_idx].add(best_j)
            else:
                fp += 1

        fn = n_gt - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {'TP': tp, 'FP': fp, 'FN': fn,
                'Precision': precision, 'Recall': recall, 'F1': f1}

    def compute_metrics(self):
        results = {'per_class': {}, 'overall': {}, 'iou_stats': {}}

        for class_id in range(1, self.num_classes):
            class_name = self.class_names[class_id]
            results['per_class'][class_name] = {}
            for iou_thresh in self.iou_thresholds:
                prf = self._compute_prf(class_id, iou_thresh)
                prf['AP'] = self._compute_ap(class_id, iou_thresh)
                results['per_class'][class_name][f'iou_{iou_thresh}'] = prf

        for iou_thresh in self.iou_thresholds:
            total_tp = sum(results['per_class'][self.class_names[c]][f'iou_{iou_thresh}']['TP']
                           for c in range(1, self.num_classes))
            total_fp = sum(results['per_class'][self.class_names[c]][f'iou_{iou_thresh}']['FP']
                           for c in range(1, self.num_classes))
            total_fn = sum(results['per_class'][self.class_names[c]][f'iou_{iou_thresh}']['FN']
                           for c in range(1, self.num_classes))
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            results['overall'][f'iou_{iou_thresh}'] = {
                'TP': total_tp, 'FP': total_fp, 'FN': total_fn,
                'Precision': precision, 'Recall': recall, 'F1': f1
            }

        results['mAP50'] = float(np.mean([
            results['per_class'][self.class_names[c]]['iou_0.5']['AP']
            for c in range(1, self.num_classes)
        ]))

        results['mAP50_95'] = float(np.mean([
            results['per_class'][self.class_names[c]][f'iou_{t}']['AP']
            for c in range(1, self.num_classes)
            for t in self.iou_thresholds
        ]))

        results['mAP_per_class'] = {}
        for class_id in range(1, self.num_classes):
            class_name = self.class_names[class_id]
            results['mAP_per_class'][class_name] = {
                'AP50': results['per_class'][class_name]['iou_0.5']['AP'],
                'AP50_95': float(np.mean([
                    results['per_class'][class_name][f'iou_{t}']['AP']
                    for t in self.iou_thresholds
                ]))
            }

        if self.box_ious:
            results['iou_stats']['box_iou_mean'] = float(np.mean(self.box_ious))
            results['iou_stats']['box_iou_std'] = float(np.std(self.box_ious))
            results['iou_stats']['box_iou_median'] = float(np.median(self.box_ious))
        if self.mask_ious:
            results['iou_stats']['mask_iou_mean'] = float(np.mean(self.mask_ious))
            results['iou_stats']['mask_iou_std'] = float(np.std(self.mask_ious))
            results['iou_stats']['mask_iou_median'] = float(np.median(self.mask_ious))

        return results


# =============================================================================
# CHARGEMENT DES GROUND TRUTHS
# =============================================================================

def load_ground_truths(images_dir, annotations_file):
    coco = COCO(annotations_file)
    cat_ids = coco.getCatIds()
    cat_mapping = {cat_id: idx + 1 for idx, cat_id in enumerate(cat_ids)}

    ground_truths = {}

    for img_id in coco.imgs:
        img_info = coco.imgs[img_id]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        masks = []

        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(cat_mapping[ann['category_id']])

            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], list):
                    rles = coco_mask_utils.frPyObjects(
                        ann['segmentation'], img_info['height'], img_info['width'])
                    rle = coco_mask_utils.merge(rles)
                    mask = coco_mask_utils.decode(rle)
                else:
                    mask = coco_mask_utils.decode(ann['segmentation'])
                masks.append(mask)

        ground_truths[img_info['file_name']] = {
            'boxes': np.array(boxes) if boxes else np.zeros((0, 4)),
            'labels': np.array(labels) if labels else np.zeros((0,), dtype=int),
            'masks': masks,
            'image_id': img_id,
            'width': img_info['width'],
            'height': img_info['height']
        }

    return ground_truths


# =============================================================================
# INFÉRENCE (pour évaluation)
# =============================================================================

def predict_for_eval(model, image_path, gt_width, gt_height, image_size=512):
    """Prédiction + extraction d'instances pour comparaison avec GT."""
    from scipy import ndimage as ndi

    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((image_size, image_size), Image.BILINEAR)
    image_array = np.array(image_resized, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image_array = (image_array - mean) / std
    image_batch = np.expand_dims(image_array, axis=0)

    output = model.predict(image_batch, verbose=0)

    if output.ndim == 4:
        pred_mask = np.argmax(output[0], axis=-1).astype(np.uint8)
    else:
        pred_mask = output[0].astype(np.uint8)

    # Redimensionner à la taille GT
    pred_pil = Image.fromarray(pred_mask)
    pred_pil = pred_pil.resize((gt_width, gt_height), Image.NEAREST)
    pred_mask = np.array(pred_pil)

    # Extraire instances par composantes connexes
    pred_boxes = []
    pred_labels = []
    pred_masks = []

    classes = CONFIG["classes"]
    for class_id in range(1, len(classes)):
        binary_mask = (pred_mask == class_id).astype(np.uint8)
        labeled_array, num_features = ndi.label(binary_mask)

        for i in range(1, num_features + 1):
            instance_mask = (labeled_array == i).astype(np.uint8)
            if instance_mask.sum() < 100:
                continue
            rows = np.any(instance_mask, axis=1)
            cols = np.any(instance_mask, axis=0)
            if not rows.any() or not cols.any():
                continue
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]
            pred_boxes.append([float(x1), float(y1), float(x2 + 1), float(y2 + 1)])
            pred_labels.append(class_id)
            pred_masks.append(instance_mask > 0)

    pred_boxes_arr = np.array(pred_boxes) if pred_boxes else np.zeros((0, 4))
    pred_labels_arr = np.array(pred_labels, dtype=int) if pred_labels else np.zeros((0,), dtype=int)

    return pred_boxes_arr, pred_labels_arr, pred_masks


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_metrics(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    class_names = list(results['mAP_per_class'].keys())
    ap50_values = [results['mAP_per_class'][c]['AP50'] for c in class_names]
    ap50_95_values = [results['mAP_per_class'][c]['AP50_95'] for c in class_names]

    x = np.arange(len(class_names))
    width = 0.35

    axes[0].bar(x - width/2, ap50_values, width, label='AP@50', color='steelblue')
    axes[0].bar(x + width/2, ap50_95_values, width, label='AP@50:95', color='coral')
    axes[0].set_xlabel('Classes')
    axes[0].set_ylabel('Average Precision')
    axes[0].set_title('AP par classe')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    precisions = [results['per_class'][c]['iou_0.5']['Precision'] for c in class_names]
    recalls    = [results['per_class'][c]['iou_0.5']['Recall']    for c in class_names]
    f1s        = [results['per_class'][c]['iou_0.5']['F1']        for c in class_names]

    width = 0.25
    axes[1].bar(x - width, precisions, width, label='Precision', color='green')
    axes[1].bar(x,          recalls,   width, label='Recall',    color='blue')
    axes[1].bar(x + width,  f1s,       width, label='F1-Score',  color='red')
    axes[1].set_xlabel('Classes')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Precision / Recall / F1 par classe (IoU=0.5)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_per_class.png'), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    iou_thresholds = CONFIG['iou_thresholds']
    global_precisions = [results['overall'][f'iou_{t}']['Precision'] for t in iou_thresholds]
    global_recalls    = [results['overall'][f'iou_{t}']['Recall']    for t in iou_thresholds]
    global_f1s        = [results['overall'][f'iou_{t}']['F1']        for t in iou_thresholds]

    ax.plot(iou_thresholds, global_precisions, 'o-', label='Precision', linewidth=2, markersize=8)
    ax.plot(iou_thresholds, global_recalls,    's-', label='Recall',    linewidth=2, markersize=8)
    ax.plot(iou_thresholds, global_f1s,        '^-', label='F1-Score',  linewidth=2, markersize=8)
    ax.set_xlabel('Seuil IoU')
    ax.set_ylabel('Score')
    ax.set_title('Métriques globales vs Seuil IoU')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.set_xlim(0.45, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_vs_iou.png'), dpi=150)
    plt.close()

    print(f"📊 Graphiques sauvegardés dans: {output_dir}")


def generate_report(results, output_dir):
    report_path = os.path.join(output_dir, 'evaluation_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("   RAPPORT D'ÉVALUATION - KERAS-SEG CADASTRAL\n")
        f.write("=" * 70 + "\n")
        f.write(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("📊 RÉSUMÉ DES MÉTRIQUES PRINCIPALES\n")
        f.write("-" * 50 + "\n")
        f.write(f"   mAP@50:        {results['mAP50']:.4f} ({results['mAP50']*100:.2f}%)\n")
        f.write(f"   mAP@50:95:     {results['mAP50_95']:.4f} ({results['mAP50_95']*100:.2f}%)\n")
        f.write(f"\n   Precision@50:  {results['overall']['iou_0.5']['Precision']:.4f}\n")
        f.write(f"   Recall@50:     {results['overall']['iou_0.5']['Recall']:.4f}\n")
        f.write(f"   F1-Score@50:   {results['overall']['iou_0.5']['F1']:.4f}\n")

        if results.get('iou_stats'):
            f.write(f"\n   IoU moyen (boîtes):  {results['iou_stats'].get('box_iou_mean', 0):.4f}\n")
            f.write(f"   IoU moyen (masques): {results['iou_stats'].get('mask_iou_mean', 0):.4f}\n")

        f.write("\n\n📋 MÉTRIQUES PAR CLASSE (IoU=0.5)\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Classe':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AP50':>10}\n")
        f.write("-" * 65 + "\n")

        for class_name in results['per_class']:
            metrics = results['per_class'][class_name]['iou_0.5']
            ap50 = results['mAP_per_class'][class_name]['AP50']
            f.write(f"{class_name:<25} {metrics['Precision']:>10.4f} {metrics['Recall']:>10.4f} "
                    f"{metrics['F1']:>10.4f} {ap50:>10.4f}\n")

        f.write("\n\n📈 DÉTAILS TP/FP/FN PAR CLASSE (IoU=0.5)\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Classe':<25} {'TP':>8} {'FP':>8} {'FN':>8}\n")
        f.write("-" * 50 + "\n")

        for class_name in results['per_class']:
            metrics = results['per_class'][class_name]['iou_0.5']
            f.write(f"{class_name:<25} {metrics['TP']:>8} {metrics['FP']:>8} {metrics['FN']:>8}\n")

        f.write("\n" + "=" * 70 + "\n")

    print(f"📄 Rapport sauvegardé: {report_path}")
    return report_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("   ÉVALUATION KERAS-SEG - Segmentation des Toitures")
    print("   (Métriques identiques à Mask R-CNN, DeepLabV3+ et YOLO26-seg)")
    print("=" * 70)

    gpus = tf.config.list_physical_devices('GPU')
    device_info = f"GPU ({len(gpus)})" if gpus else "CPU"
    print(f"\n📱 Device: {device_info}")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print("\n📂 Chargement des ground truths...")
    ground_truths = load_ground_truths(
        CONFIG["images_dir"],
        CONFIG["annotations_file"]
    )
    print(f"   {len(ground_truths)} images chargées")

    print("\n🧠 Chargement du modèle...")
    model = tf.keras.models.load_model(CONFIG["model_path"], compile=False)
    print(f"✅ Modèle chargé: {CONFIG['model_path']}")

    num_classes = len(CONFIG["classes"])
    metrics_calc = MetricsCalculator(
        num_classes=num_classes,
        class_names=CONFIG["classes"],
        iou_thresholds=CONFIG["iou_thresholds"]
    )

    print("\n📊 Calcul des métriques...")

    image_files = list(ground_truths.keys())

    for img_file in tqdm(image_files, desc="Évaluation"):
        img_path = os.path.join(CONFIG["images_dir"], img_file)

        if not os.path.exists(img_path):
            continue

        gt = ground_truths[img_file]

        pred_boxes, pred_labels, pred_masks = predict_for_eval(
            model, img_path, gt['width'], gt['height'], CONFIG["image_size"]
        )

        metrics_calc.add_image(
            pred_boxes, pred_labels, pred_masks,
            gt['boxes'], gt['labels'], gt['masks']
        )

    results = metrics_calc.compute_metrics()

    print("\n" + "=" * 70)
    print("   📊 RÉSULTATS DE L'ÉVALUATION")
    print("=" * 70)

    print(f"\n🎯 MÉTRIQUES PRINCIPALES")
    print(f"   {'─' * 40}")
    print(f"   mAP@50:        {results['mAP50']:.4f} ({results['mAP50']*100:.2f}%)")
    print(f"   mAP@50:95:     {results['mAP50_95']:.4f} ({results['mAP50_95']*100:.2f}%)")
    print(f"\n   Precision@50:  {results['overall']['iou_0.5']['Precision']:.4f}")
    print(f"   Recall@50:     {results['overall']['iou_0.5']['Recall']:.4f}")
    print(f"   F1-Score@50:   {results['overall']['iou_0.5']['F1']:.4f}")

    if results.get('iou_stats'):
        print(f"\n   IoU moyen (boîtes):  {results['iou_stats'].get('box_iou_mean', 0):.4f}")
        print(f"   IoU moyen (masques): {results['iou_stats'].get('mask_iou_mean', 0):.4f}")

    print(f"\n📋 PAR CLASSE (IoU=0.5)")
    print(f"   {'─' * 40}")
    for class_name in results['per_class']:
        metrics = results['per_class'][class_name]['iou_0.5']
        print(f"   {class_name}:")
        print(f"      Precision: {metrics['Precision']:.4f} | Recall: {metrics['Recall']:.4f} | F1: {metrics['F1']:.4f}")

    results_path = os.path.join(CONFIG["output_dir"], "metrics.json")

    def convert_to_serializable(obj):
        if isinstance(obj, defaultdict):
            return dict(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_serializable = json.loads(
        json.dumps(results, default=convert_to_serializable)
    )

    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\n💾 Métriques sauvegardées: {results_path}")

    plot_metrics(results, CONFIG["output_dir"])
    generate_report(results, CONFIG["output_dir"])

    print("\n" + "=" * 70)
    print("   ✅ ÉVALUATION TERMINÉE")
    print("=" * 70)


if __name__ == "__main__":
    main()
