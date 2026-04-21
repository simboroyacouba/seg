"""
Transfer Learning - Adaptation du modèle Keras (maladies cutanées) aux toitures cadastrales
Architecture: MobileNetV2 + décodeur U-Net → segmentation multi-classes

Stratégie en 2 phases :
  Phase 1 : Geler encodeur + décodeur, entraîner uniquement la nouvelle couche de sortie
  Phase 2 : Dégeler tout, fine-tuner avec un LR faible

Dataset: Images aériennes annotées avec CVAT (format COCO)
Classes: toiture_tole_ondulee, toiture_tole_bac, toiture_dalle
"""

import os
import json
import time
import random
import warnings
import yaml
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask_utils
import tensorflow as tf
from tensorflow import keras
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_classes(yaml_path=None):
    path = yaml_path or os.getenv("CLASSES_FILE", "classes.yaml")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)['classes']

CONFIG = {
    # Chemins
    "pretrained_model_path": os.getenv("SEGMENTATION_PRETRAINED_MODEL", "./final_segmentation_model.keras"),
    "images_dir":            os.getenv("SEGMENTATION_DATASET_IMAGES_DIR"),
    "annotations_file":      os.getenv("SEGMENTATION_DATASET_ANNOTATIONS_FILE"),
    "classes_file":          os.getenv("CLASSES_FILE", "classes.yaml"),
    "output_dir":            "./output",

    # Classes
    "classes": load_classes(),

    # Image
    "image_size": 224,

    # Phase 1 : tête seule
    "phase1_epochs":    15,
    "phase1_lr":        1e-3,

    # Phase 2 : fine-tuning complet
    "phase2_epochs":    50,
    "phase2_lr":        1e-5,

    # Dataset
    "batch_size":   8,
    "train_split":  0.85,

    # Sauvegarde
    "save_every":   5,
}


# =============================================================================
# UTILITAIRES TEMPS
# =============================================================================

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds//60)}m {int(seconds%60)}s"
    else:
        return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m {int(seconds%60)}s"


class TrainingTimer:
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
        self.start_time = None
        self.epoch_times = []
        self.epoch_start = None

    def start_training(self):
        self.start_time = time.time()
        self.training_start_datetime = datetime.now()

    def start_epoch(self):
        self.epoch_start = time.time()

    def end_epoch(self, epoch):
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        total_elapsed = time.time() - self.start_time
        avg_epoch_time = np.mean(self.epoch_times)
        remaining_epochs = self.num_epochs - (epoch + 1)
        estimated_remaining = avg_epoch_time * remaining_epochs
        eta = datetime.now() + timedelta(seconds=estimated_remaining)
        return {
            'epoch_time': epoch_time,
            'total_elapsed': total_elapsed,
            'avg_epoch_time': avg_epoch_time,
            'estimated_remaining': estimated_remaining,
            'eta': eta,
            'progress_percent': ((epoch + 1) / self.num_epochs) * 100
        }

    def get_final_stats(self):
        total_time = time.time() - self.start_time
        return {
            'total_time': total_time,
            'total_time_formatted': format_time(total_time),
            'avg_epoch_time': np.mean(self.epoch_times),
            'avg_epoch_time_formatted': format_time(np.mean(self.epoch_times)),
            'min_epoch_time': float(np.min(self.epoch_times)),
            'min_epoch_time_formatted': format_time(np.min(self.epoch_times)),
            'max_epoch_time': float(np.max(self.epoch_times)),
            'max_epoch_time_formatted': format_time(np.max(self.epoch_times)),
            'std_epoch_time': float(np.std(self.epoch_times)),
            'epoch_times': [float(t) for t in self.epoch_times],
            'start_datetime': self.training_start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            'end_datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


# =============================================================================
# DATASET KERAS (tf.data)
# =============================================================================

class COCOSegDataset:
    """
    Dataset COCO → masques sémantiques pour TensorFlow.
    Même logique que DeepLabV3+ et Mask R-CNN.
    """

    def __init__(self, images_dir, annotations_file, image_size=224):
        self.images_dir = images_dir
        self.image_size = image_size

        self.coco = COCO(annotations_file)
        self.image_ids = list(self.coco.imgs.keys())

        cat_ids = self.coco.getCatIds()
        self.cat_mapping = {cat_id: idx + 1 for idx, cat_id in enumerate(cat_ids)}

        print(f"Dataset chargé: {len(self.image_ids)} images")
        print(f"Catégories: {[self.coco.cats[c]['name'] for c in cat_ids]}")

    def __len__(self):
        return len(self.image_ids)

    def load_sample(self, img_id):
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.images_dir, img_info['file_name'])

        image = Image.open(img_path).convert("RGB")
        mask  = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        for ann in self.coco.loadAnns(ann_ids):
            if ann.get('iscrowd', 0):
                continue
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            class_id = self.cat_mapping[ann['category_id']]
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], list):
                    rles = coco_mask_utils.frPyObjects(
                        ann['segmentation'], img_info['height'], img_info['width'])
                    rle  = coco_mask_utils.merge(rles)
                    m    = coco_mask_utils.decode(rle)
                else:
                    m = coco_mask_utils.decode(ann['segmentation'])
                mask[m > 0] = class_id

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask  = np.array(Image.fromarray(mask).resize(
            (self.image_size, self.image_size), Image.NEAREST))

        return np.array(image, dtype=np.float32), mask.astype(np.int32)

    def split(self, train_ratio=0.85, seed=42):
        random.seed(seed)
        ids = self.image_ids.copy()
        random.shuffle(ids)
        split = int(len(ids) * train_ratio)
        return ids[:split], ids[split:]


def augment(image, mask):
    """Augmentation identique aux autres modèles (flip H/V)."""
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask  = tf.reverse(mask, axis=[1])
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask  = tf.reverse(mask, axis=[0])
    return image, mask


def normalize(image, mask):
    """Normalisation ImageNet."""
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    image = (image / 255.0 - mean) / std
    return image, mask


def build_tf_dataset(dataset_obj, image_ids, batch_size, train=True):
    """Construire un tf.data.Dataset à partir de la liste d'IDs."""

    def generator():
        for img_id in image_ids:
            img, mask = dataset_obj.load_sample(img_id)
            yield img, mask

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(dataset_obj.image_size, dataset_obj.image_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(dataset_obj.image_size, dataset_obj.image_size),   dtype=tf.int32),
        )
    )

    if train:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# =============================================================================
# MODÈLE : TRANSFER LEARNING
# =============================================================================

def build_transfer_model(pretrained_path, num_classes):
    """
    Charger le modèle pré-entraîné (maladies cutanées),
    remplacer la sortie binaire par une sortie multi-classes.
    """
    base_model = tf.keras.models.load_model(pretrained_path, compile=False)

    # Trouver la couche juste avant la sortie binaire
    output_layer_name = "segmentation_output"
    last_conv_layer = None
    for layer in reversed(base_model.layers):
        if layer.name == output_layer_name:
            continue
        if hasattr(layer, 'output'):
            last_conv_layer = layer
            break

    # Construire le nouveau modèle avec la sortie multi-classes
    x = last_conv_layer.output

    # Nouvelle tête : Conv2D(num_classes, softmax)
    new_output = tf.keras.layers.Conv2D(
        filters=num_classes,
        kernel_size=1,
        activation='softmax',
        name='segmentation_output_multiclass',
        padding='same'
    )(x)

    new_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=new_output,
        name='unet_mobilenetv2_cadastral'
    )

    return new_model, base_model


def freeze_base(model, base_model):
    """Geler toutes les couches sauf la nouvelle tête (Phase 1)."""
    new_head_names = {'segmentation_output_multiclass'}
    for layer in model.layers:
        layer.trainable = layer.name in new_head_names
    trainable = sum(1 for l in model.layers if l.trainable)
    print(f"   Couches entraînables : {trainable}/{len(model.layers)}")


def unfreeze_all(model):
    """Dégeler toutes les couches (Phase 2)."""
    for layer in model.layers:
        layer.trainable = True
    trainable = sum(1 for l in model.layers if l.trainable)
    print(f"   Couches entraînables : {trainable}/{len(model.layers)}")


# =============================================================================
# LOSS & MÉTRIQUES
# =============================================================================

def sparse_dice_loss(y_true, y_pred, num_classes, smooth=1e-6):
    """Dice loss pour segmentation multi-classes."""
    y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
    y_true_oh = tf.cast(y_true_oh, tf.float32)
    intersection = tf.reduce_sum(y_true_oh * y_pred, axis=[1, 2])
    union        = tf.reduce_sum(y_true_oh + y_pred,  axis=[1, 2])
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - tf.reduce_mean(dice)


def combined_loss(num_classes):
    """CrossEntropy + Dice Loss (standard pour segmentation médicale)."""
    ce = tf.keras.losses.SparseCategoricalCrossentropy()

    def loss_fn(y_true, y_pred):
        ce_loss   = ce(y_true, y_pred)
        dice_loss = sparse_dice_loss(y_true, y_pred, num_classes)
        return ce_loss + dice_loss

    loss_fn.__name__ = 'combined_loss'
    return loss_fn


def mean_iou_metric(num_classes):
    return tf.keras.metrics.MeanIoU(num_classes=num_classes, name='mean_iou')


# =============================================================================
# BOUCLE D'ENTRAÎNEMENT MANUELLE (pour suivi identique aux autres modèles)
# =============================================================================

def train_epoch(model, ds, optimizer, loss_fn, num_classes):
    total_loss = 0.0
    total_iou  = 0.0
    n_batches  = 0
    iou_metric = tf.keras.metrics.MeanIoU(num_classes=num_classes)

    for images, masks in ds:
        with tf.GradientTape() as tape:
            preds = model(images, training=True)
            loss  = loss_fn(masks, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        pred_classes = tf.argmax(preds, axis=-1, output_type=tf.int32)
        iou_metric.update_state(masks, pred_classes)

        total_loss += loss.numpy()
        n_batches  += 1

    return float(total_loss / n_batches), float(iou_metric.result())


def val_epoch(model, ds, loss_fn, num_classes):
    total_loss = 0.0
    n_batches  = 0
    iou_metric = tf.keras.metrics.MeanIoU(num_classes=num_classes)

    for images, masks in ds:
        preds = model(images, training=False)
        loss  = loss_fn(masks, preds)

        pred_classes = tf.argmax(preds, axis=-1, output_type=tf.int32)
        iou_metric.update_state(masks, pred_classes)

        total_loss += loss.numpy()
        n_batches  += 1

    return float(total_loss / n_batches), float(iou_metric.result())


def run_phase(phase_num, model, train_ds, val_ds, optimizer, loss_fn,
              num_classes, num_epochs, output_dir, history, timer,
              best_val_loss, save_every):
    print(f"\n{'='*70}")
    print(f"   🔁 PHASE {phase_num}")
    print(f"{'='*70}")

    for epoch in range(num_epochs):
        timer.start_epoch()
        global_epoch = len(history['train_loss'])

        train_loss, train_iou = train_epoch(model, train_ds, optimizer, loss_fn, num_classes)
        val_loss,   val_iou   = val_epoch(model, val_ds, loss_fn, num_classes)

        time_stats = timer.end_epoch(epoch)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['lr'].append(float(optimizer.learning_rate))
        history['epoch_times'].append(time_stats['epoch_time'])
        history['cumulative_times'].append(time_stats['total_elapsed'])

        print(f"\n{'─'*70}")
        print(f"📈 Phase {phase_num} | Epoch {epoch+1}/{num_epochs} | "
              f"Global {global_epoch+1} | {time_stats['progress_percent']:.1f}%")
        print(f"{'─'*70}")
        print(f"   📉 Train Loss: {train_loss:.4f}  |  IoU: {train_iou:.4f}")
        print(f"   📊 Val Loss:   {val_loss:.4f}  |  IoU: {val_iou:.4f}")
        print(f"   📐 LR:         {optimizer.learning_rate.numpy():.2e}")
        print(f"{'─'*70}")
        print(f"   ⏱️  Temps epoch:       {format_time(time_stats['epoch_time'])}")
        print(f"   ⏱️  Temps total:       {format_time(time_stats['total_elapsed'])}")
        print(f"   ⏳ Temps restant:      {format_time(time_stats['estimated_remaining'])}")
        print(f"   🏁 ETA:                {time_stats['eta'].strftime('%H:%M:%S')}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(os.path.join(output_dir, "best_model.keras"))
            print(f"   ✅ Meilleur modèle sauvegardé! (val_loss={val_loss:.4f})")

        if (global_epoch + 1) % save_every == 0:
            model.save(os.path.join(output_dir, f"checkpoint_epoch_{global_epoch+1}.keras"))
            print(f"   💾 Checkpoint epoch {global_epoch+1} sauvegardé")

    return best_val_loss


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("   Keras U-Net+MobileNetV2 - Transfer Learning Toitures Cadastrales")
    print("   (Adaptation depuis détection de maladies cutanées)")
    print("=" * 70)

    gpus = tf.config.list_physical_devices('GPU')
    device_info = f"GPU ({len(gpus)})" if gpus else "CPU"
    print(f"\n📱 Device: {device_info}")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    num_classes = len(CONFIG["classes"])

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("\n📂 Chargement du dataset...")
    dataset = COCOSegDataset(
        CONFIG["images_dir"],
        CONFIG["annotations_file"],
        image_size=CONFIG["image_size"]
    )

    train_ids, val_ids = dataset.split(CONFIG["train_split"])
    print(f"   Train: {len(train_ids)} images")
    print(f"   Val:   {len(val_ids)} images")

    train_ds = build_tf_dataset(dataset, train_ids, CONFIG["batch_size"], train=True)
    val_ds   = build_tf_dataset(dataset, val_ids,   CONFIG["batch_size"], train=False)

    # ── Modèle ───────────────────────────────────────────────────────────────
    print(f"\n🧠 Chargement du modèle pré-entraîné...")
    print(f"   Source: {CONFIG['pretrained_model_path']}")
    model, base_model = build_transfer_model(
        CONFIG["pretrained_model_path"], num_classes)

    print(f"\n   Architecture: {model.name}")
    print(f"   Couches totales: {len(model.layers)}")
    print(f"   Paramètres: {model.count_params():,}")
    print(f"   Classes: {CONFIG['classes']}")

    loss_fn = combined_loss(num_classes)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou':  [], 'val_iou':  [],
        'lr': [], 'epoch_times': [], 'cumulative_times': []
    }

    best_val_loss = float('inf')
    total_epochs  = CONFIG["phase1_epochs"] + CONFIG["phase2_epochs"]
    timer = TrainingTimer(total_epochs)
    timer.start_training()

    # ── Phase 1 : tête seule ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"   ❄️  PHASE 1 : Encodeur + décodeur gelés — entraînement de la tête")
    print(f"   Epochs: {CONFIG['phase1_epochs']} | LR: {CONFIG['phase1_lr']}")
    freeze_base(model, base_model)

    optimizer1 = tf.keras.optimizers.Adam(learning_rate=CONFIG["phase1_lr"])

    # Réinitialiser le timer pour la phase 1
    timer1 = TrainingTimer(CONFIG["phase1_epochs"])
    timer1.start_training()

    best_val_loss = run_phase(
        phase_num=1,
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        optimizer=optimizer1,
        loss_fn=loss_fn,
        num_classes=num_classes,
        num_epochs=CONFIG["phase1_epochs"],
        output_dir=CONFIG["output_dir"],
        history=history,
        timer=timer1,
        best_val_loss=best_val_loss,
        save_every=CONFIG["save_every"]
    )

    # ── Phase 2 : fine-tuning complet ────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"   🔥 PHASE 2 : Fine-tuning complet (LR faible)")
    print(f"   Epochs: {CONFIG['phase2_epochs']} | LR: {CONFIG['phase2_lr']}")
    unfreeze_all(model)

    optimizer2 = tf.keras.optimizers.Adam(learning_rate=CONFIG["phase2_lr"])

    timer2 = TrainingTimer(CONFIG["phase2_epochs"])
    timer2.start_training()

    best_val_loss = run_phase(
        phase_num=2,
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        optimizer=optimizer2,
        loss_fn=loss_fn,
        num_classes=num_classes,
        num_epochs=CONFIG["phase2_epochs"],
        output_dir=CONFIG["output_dir"],
        history=history,
        timer=timer2,
        best_val_loss=best_val_loss,
        save_every=CONFIG["save_every"]
    )

    # ── Modèle final ─────────────────────────────────────────────────────────
    model.save(os.path.join(CONFIG["output_dir"], "final_model.keras"))

    # ── Stats finales ─────────────────────────────────────────────────────────
    all_epoch_times = timer1.epoch_times + timer2.epoch_times
    total_time = sum(all_epoch_times)
    final_time_stats = {
        'total_time': total_time,
        'total_time_formatted': format_time(total_time),
        'avg_epoch_time': float(np.mean(all_epoch_times)),
        'avg_epoch_time_formatted': format_time(np.mean(all_epoch_times)),
        'min_epoch_time_formatted': format_time(np.min(all_epoch_times)),
        'max_epoch_time_formatted': format_time(np.max(all_epoch_times)),
        'std_epoch_time': float(np.std(all_epoch_times)),
        'epoch_times': [float(t) for t in all_epoch_times],
        'start_datetime': timer1.training_start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        'end_datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    history['time_stats'] = final_time_stats

    # ── Historique JSON ───────────────────────────────────────────────────────
    with open(os.path.join(CONFIG["output_dir"], "history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    # ── Graphiques ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs_range = range(1, len(history['train_loss']) + 1)
    phase1_end = CONFIG["phase1_epochs"]

    axes[0].plot(epochs_range, history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(epochs_range, history['val_loss'],   label='Validation', linewidth=2)
    axes[0].axvline(x=phase1_end + 0.5, color='gray', linestyle='--', label='Phase 1→2')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Courbes de perte — Keras U-Net Cadastral')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, history['train_iou'], label='Train IoU', linewidth=2)
    axes[1].plot(epochs_range, history['val_iou'],   label='Val IoU',   linewidth=2)
    axes[1].axvline(x=phase1_end + 0.5, color='gray', linestyle='--', label='Phase 1→2')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Mean IoU')
    axes[1].set_title('Mean IoU par epoch')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].bar(epochs_range, history['epoch_times'], color='steelblue', alpha=0.7)
    axes[2].axhline(y=final_time_stats['avg_epoch_time'], color='red', linestyle='--',
                    label=f"Moyenne: {final_time_stats['avg_epoch_time_formatted']}")
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Temps (s)')
    axes[2].set_title('Temps par epoch')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "training_curves.png"), dpi=150)
    plt.close()

    # ── Rapport final ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("   🎉 ENTRAÎNEMENT TERMINÉ")
    print("=" * 70)
    print(f"\n📊 RÉSUMÉ DES PERFORMANCES")
    print(f"   {'─'*50}")
    print(f"   Meilleure Val Loss: {best_val_loss:.4f}")
    print(f"   Val Loss finale:    {history['val_loss'][-1]:.4f}")
    print(f"   Val IoU finale:     {history['val_iou'][-1]:.4f}")
    print(f"\n⏱️  RAPPORT DE TEMPS")
    print(f"   {'─'*50}")
    print(f"   Début:              {final_time_stats['start_datetime']}")
    print(f"   Fin:                {final_time_stats['end_datetime']}")
    print(f"   Temps total:        {final_time_stats['total_time_formatted']}")
    print(f"   Temps moyen/epoch:  {final_time_stats['avg_epoch_time_formatted']}")
    print(f"\n💾 FICHIERS SAUVEGARDÉS")
    print(f"   {'─'*50}")
    print(f"   📁 Dossier: {CONFIG['output_dir']}")
    print(f"   ├── best_model.keras")
    print(f"   ├── final_model.keras")
    print(f"   ├── checkpoint_epoch_*.keras")
    print(f"   ├── history.json")
    print(f"   └── training_curves.png")
    print("=" * 70)

    # Rapport texte
    report_path = os.path.join(CONFIG["output_dir"], "training_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("   RAPPORT D'ENTRAÎNEMENT - KERAS U-NET CADASTRAL\n")
        f.write("   (Transfer Learning depuis détection maladies cutanées)\n")
        f.write("=" * 70 + "\n\n")
        f.write("CONFIGURATION\n" + "-" * 50 + "\n")
        for key, value in CONFIG.items():
            f.write(f"   {key}: {value}\n")
        f.write("\nSTRATÉGIE TRANSFER LEARNING\n" + "-" * 50 + "\n")
        f.write(f"   Modèle source:    {CONFIG['pretrained_model_path']}\n")
        f.write(f"   Phase 1 ({CONFIG['phase1_epochs']} epochs): encodeur + décodeur gelés, LR={CONFIG['phase1_lr']}\n")
        f.write(f"   Phase 2 ({CONFIG['phase2_epochs']} epochs): fine-tuning complet,        LR={CONFIG['phase2_lr']}\n")
        f.write(f"   Couche remplacée: Conv2D(1, sigmoid) → Conv2D({num_classes}, softmax)\n")
        f.write("\nPERFORMANCES\n" + "-" * 50 + "\n")
        f.write(f"   Meilleure Val Loss: {best_val_loss:.4f}\n")
        f.write(f"   Val Loss finale:    {history['val_loss'][-1]:.4f}\n")
        f.write(f"   Val IoU finale:     {history['val_iou'][-1]:.4f}\n")
        f.write("\nTEMPS D'ENTRAÎNEMENT\n" + "-" * 50 + "\n")
        f.write(f"   Début:              {final_time_stats['start_datetime']}\n")
        f.write(f"   Fin:                {final_time_stats['end_datetime']}\n")
        f.write(f"   Temps total:        {final_time_stats['total_time_formatted']}\n")
        f.write(f"   Temps moyen/epoch:  {final_time_stats['avg_epoch_time_formatted']}\n")
        f.write(f"   Écart-type:         {final_time_stats['std_epoch_time']:.2f}s\n")
        f.write("\nTEMPS PAR EPOCH\n" + "-" * 50 + "\n")
        for i, t in enumerate(final_time_stats['epoch_times']):
            phase = 1 if i < CONFIG["phase1_epochs"] else 2
            f.write(f"   Epoch {i+1:3d} (Ph{phase}): {format_time(t)}\n")

    print(f"\n📄 Rapport sauvegardé: {report_path}")


if __name__ == "__main__":
    main()
