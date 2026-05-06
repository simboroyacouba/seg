"""
Transfer Learning - Adaptation du modèle Keras (maladies cutanées) aux toitures cadastrales
Architecture: MobileNetV2 + décodeur U-Net → segmentation multi-classes

Modes d'entraînement:
  simple    : Transfer learning standard (Phase 1 tête + Phase 2 fine-tuning)
  attention : Transfer learning + CBAM Keras injecté avant la tête de sortie
  optimize  : Transfer learning standard + recherche bayésienne des hyperparamètres (Optuna)
"""

import os
import json
import time
import random
import argparse
import warnings
import yaml
import numpy as np
from datetime import datetime, timedelta
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask_utils
import tensorflow as tf
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_classes(yaml_path=None):
    path = yaml_path or os.getenv("CLASSES_FILE", "classes.yaml")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)['classes']

OPTUNA_CONFIG = {
    "n_trials":          20,
    "n_epochs_per_trial": 6,   # 3 epochs phase1 + 3 epochs phase2
    "study_name":        "segperso_cadastral",
    "output_dir":        "./optuna_output",
}

CONFIG = {
    "pretrained_model_path": os.getenv("SEGMENTATION_PRETRAINED_MODEL", "./final_segmentation_model.keras"),
    "images_dir":            os.getenv("SEGMENTATION_DATASET_IMAGES_DIR"),
    "annotations_file":      os.getenv("SEGMENTATION_DATASET_ANNOTATIONS_FILE"),
    "classes_file":          os.getenv("CLASSES_FILE", "classes.yaml"),
    "output_dir":            "./output",
    "classes":               load_classes(),
    "image_size":            224,
    # Phase 1 : tête seule
    "phase1_epochs": 15,
    "phase1_lr":     1e-3,
    # Phase 2 : fine-tuning complet
    "phase2_epochs": 50,
    "phase2_lr":     1e-5,
    # Dataset
    "batch_size":  8,
    "train_split": 0.85,
    # Sauvegarde
    "save_every": 5,
    # CBAM (mode attention uniquement)
    "cbam_reduction":   16,
    "cbam_kernel_size": 7,
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
        self.num_epochs  = num_epochs
        self.start_time  = None
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
        total_elapsed       = time.time() - self.start_time
        avg_epoch_time      = np.mean(self.epoch_times)
        estimated_remaining = avg_epoch_time * (self.num_epochs - (epoch + 1))
        return {
            'epoch_time':          epoch_time,
            'total_elapsed':       total_elapsed,
            'avg_epoch_time':      avg_epoch_time,
            'estimated_remaining': estimated_remaining,
            'eta':                 datetime.now() + timedelta(seconds=estimated_remaining),
            'progress_percent':    ((epoch + 1) / self.num_epochs) * 100,
        }

    def get_final_stats(self):
        total_time = time.time() - self.start_time
        return {
            'total_time':               total_time,
            'total_time_formatted':     format_time(total_time),
            'avg_epoch_time':           float(np.mean(self.epoch_times)),
            'avg_epoch_time_formatted': format_time(np.mean(self.epoch_times)),
            'min_epoch_time_formatted': format_time(np.min(self.epoch_times)),
            'max_epoch_time_formatted': format_time(np.max(self.epoch_times)),
            'std_epoch_time':           float(np.std(self.epoch_times)),
            'epoch_times':              [float(t) for t in self.epoch_times],
            'start_datetime':           self.training_start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            'end_datetime':             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


# =============================================================================
# DATASET (tf.data)
# =============================================================================

class COCOSegDataset:
    def __init__(self, images_dir, annotations_file, image_size=224):
        self.images_dir = images_dir
        self.image_size = image_size
        self.coco       = COCO(annotations_file)
        self.image_ids  = list(self.coco.imgs.keys())
        cat_ids = self.coco.getCatIds()
        self.cat_mapping = {cat_id: idx + 1 for idx, cat_id in enumerate(cat_ids)}
        print(f"Dataset chargé: {len(self.image_ids)} images")
        print(f"Catégories: {[self.coco.cats[c]['name'] for c in cat_ids]}")

    def __len__(self):
        return len(self.image_ids)

    def load_sample(self, img_id):
        img_info = self.coco.imgs[img_id]
        image    = Image.open(os.path.join(self.images_dir, img_info['file_name'])).convert("RGB")
        mask     = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)):
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
                    m = coco_mask_utils.decode(coco_mask_utils.merge(rles))
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

    @property
    def cat_id_to_name(self):
        cat_ids = self.coco.getCatIds()
        return {cid: self.coco.cats[cid]['name'] for cid in cat_ids}


def augment(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask  = tf.reverse(mask, axis=[1])
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask  = tf.reverse(mask, axis=[0])
    return image, mask


def normalize(image, mask):
    mean  = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std   = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    image = (image / 255.0 - mean) / std
    return image, mask


def build_tf_dataset(dataset_obj, image_ids, batch_size, train=True):
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
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# =============================================================================
# AUGMENTATION PAR CLASSE
# =============================================================================

def load_aug_coefficients(classes):
    """
    Charge les coefficients d'augmentation depuis les variables d'environnement.
    Format index  : CLASS_AUG_1=2  (1 = première classe hors __background__)
    Format nom    : CLASS_AUG_TOITURE_TOLE_BAC=2
    Valeur par défaut: 1 (aucune augmentation).
    """
    real_classes = [c for c in classes if c != '__background__']
    coeffs = {}
    for i, cls in enumerate(real_classes, 1):
        env_idx  = f"CLASS_AUG_{i}"
        env_name = "CLASS_AUG_" + cls.upper().replace(' ', '_').replace('-', '_')
        raw = os.getenv(env_name) or os.getenv(env_idx, "1")
        coeffs[cls] = max(1, int(raw))
    return coeffs


def _count_class_stats_keras(coco, image_ids, cat_id_to_name):
    stats = {name: {'images': set(), 'annotations': 0} for name in cat_id_to_name.values()}
    for img_id in image_ids:
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            if ann.get('iscrowd', 0):
                continue
            cid = ann['category_id']
            if cid in cat_id_to_name:
                name = cat_id_to_name[cid]
                stats[name]['images'].add(img_id)
                stats[name]['annotations'] += 1
    return {name: {'images': len(s['images']), 'annotations': s['annotations']}
            for name, s in stats.items()}


def oversample_by_class_keras(image_ids, coco, cat_id_to_name, aug_coeffs):
    """Duplique les IDs d'images selon le coefficient maximal de leurs classes."""
    img_max_coeff = {}
    for img_id in image_ids:
        max_c = 1
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            if ann.get('iscrowd', 0):
                continue
            cid = ann['category_id']
            if cid in cat_id_to_name:
                max_c = max(max_c, aug_coeffs.get(cat_id_to_name[cid], 1))
        img_max_coeff[img_id] = max_c
    augmented = []
    for img_id in image_ids:
        augmented.extend([img_id] * img_max_coeff[img_id])
    return augmented


def print_augmentation_report_keras(coco, train_ids_before, train_ids_after,
                                     cat_id_to_name, aug_coeffs):
    before = _count_class_stats_keras(coco, train_ids_before, cat_id_to_name)
    after  = _count_class_stats_keras(coco, train_ids_after,  cat_id_to_name)
    print(f"\n{'='*70}")
    print(f"   RAPPORT D'AUGMENTATION DES DONNEES D'ENTRAINEMENT")
    print(f"{'='*70}")
    print(f"\n   AVANT AUGMENTATION  ({len(set(train_ids_before))} images uniques en train)")
    print(f"   {'─'*65}")
    print(f"   {'Classe':<38} {'Coeff':>5}  {'Images':>7}  {'Annot.':>7}")
    print(f"   {'─'*65}")
    total_ann_b = 0
    for cls_name, s in before.items():
        coeff  = aug_coeffs.get(cls_name, 1)
        marker = "  *" if coeff > 1 else ""
        print(f"   {cls_name:<38} x{coeff:>4}  {s['images']:>7}  {s['annotations']:>7}{marker}")
        total_ann_b += s['annotations']
    print(f"   {'─'*65}")
    print(f"   {'TOTAL':<38}       {len(set(train_ids_before)):>7}  {total_ann_b:>7}")
    print(f"\n   APRES AUGMENTATION  ({len(train_ids_after)} samples d'entrainement)")
    print(f"   {'─'*65}")
    print(f"   {'Classe':<38} {'Delta':>5}  {'Images':>7}  {'Annot.':>7}")
    print(f"   {'─'*65}")
    total_ann_a = 0
    for cls_name, sa in after.items():
        sb    = before[cls_name]
        delta = f"+{sa['images'] - sb['images']}" if sa['images'] != sb['images'] else "  ="
        print(f"   {cls_name:<38} {delta:>5}  {sa['images']:>7}  {sa['annotations']:>7}")
        total_ann_a += sa['annotations']
    print(f"   {'─'*65}")
    print(f"   {'TOTAL (avec duplicats)':<38}       {len(train_ids_after):>7}  {total_ann_a:>7}")
    ratio = len(train_ids_after) / max(len(train_ids_before), 1)
    print(f"\n   Ratio d'augmentation global: x{ratio:.2f} samples")
    print(f"{'='*70}\n")


# =============================================================================
# MÉCANISME D'ATTENTION CBAM (Keras) — mode "attention" uniquement
# =============================================================================

class ChannelAttention(tf.keras.layers.Layer):
    """Squeeze-and-Excitation channel attention (Keras)."""

    def __init__(self, channels, reduction=16, **kwargs):
        super().__init__(**kwargs)
        mid = max(channels // reduction, 8)
        self.dense1 = tf.keras.layers.Dense(mid, use_bias=False, activation='relu')
        self.dense2 = tf.keras.layers.Dense(channels, use_bias=False)

    def call(self, x):
        avg   = tf.reduce_mean(x, axis=[1, 2])
        mx    = tf.reduce_max(x,  axis=[1, 2])
        scale = tf.sigmoid(self.dense2(self.dense1(avg)) + self.dense2(self.dense1(mx)))
        return x * scale[:, tf.newaxis, tf.newaxis, :]


class SpatialAttention(tf.keras.layers.Layer):
    """Spatial attention via channel-pooled convolution (Keras)."""

    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(
            1, kernel_size, padding='same', use_bias=False, activation='sigmoid'
        )

    def call(self, x):
        avg    = tf.reduce_mean(x, axis=-1, keepdims=True)
        mx     = tf.reduce_max(x,  axis=-1, keepdims=True)
        return x * self.conv(tf.concat([avg, mx], axis=-1))


class CBAMLayer(tf.keras.layers.Layer):
    """Convolutional Block Attention Module pour Keras."""

    def __init__(self, channels, reduction=16, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def call(self, x):
        return self.spatial_att(self.channel_att(x))


# =============================================================================
# MODÈLE
# =============================================================================

def _get_last_conv_layer(base_model):
    """Retourne la couche juste avant la couche de sortie binaire d'origine."""
    for layer in reversed(base_model.layers):
        if layer.name == "segmentation_output":
            continue
        if hasattr(layer, 'output'):
            return layer
    raise ValueError("Impossible de trouver la couche précédant la sortie.")


def build_model_simple(pretrained_path, num_classes):
    """Transfer learning standard: remplace la sortie binaire par une sortie multi-classes."""
    base_model       = tf.keras.models.load_model(pretrained_path, compile=False)
    last_conv_layer  = _get_last_conv_layer(base_model)
    x                = last_conv_layer.output
    new_output = tf.keras.layers.Conv2D(
        filters=num_classes, kernel_size=1, activation='softmax',
        name='segmentation_output_multiclass', padding='same'
    )(x)
    model = tf.keras.Model(
        inputs=base_model.input, outputs=new_output,
        name='unet_mobilenetv2_cadastral'
    )
    return model, base_model


def build_model_attention(pretrained_path, num_classes, cbam_reduction=16, cbam_kernel_size=7):
    """
    Transfer learning + CBAM(Keras) injecté avant la tête de sortie.
    Structure: last_conv → CBAM → Conv2D(num_classes, softmax)
    """
    base_model      = tf.keras.models.load_model(pretrained_path, compile=False)
    last_conv_layer = _get_last_conv_layer(base_model)
    x               = last_conv_layer.output

    channels = x.shape[-1]
    if channels is None:
        print("   [AVERTISSEMENT] Channel count inconnu, CBAM utilise 32 channels par defaut.")
        channels = 32

    x = CBAMLayer(channels, cbam_reduction, cbam_kernel_size, name='cbam_attention')(x)
    new_output = tf.keras.layers.Conv2D(
        filters=num_classes, kernel_size=1, activation='softmax',
        name='segmentation_output_multiclass', padding='same'
    )(x)
    model = tf.keras.Model(
        inputs=base_model.input, outputs=new_output,
        name='unet_mobilenetv2_cbam_cadastral'
    )
    return model, base_model


def freeze_base(model, trainable_layer_names):
    """Gèle toutes les couches sauf celles dont le nom est dans trainable_layer_names."""
    for layer in model.layers:
        layer.trainable = layer.name in trainable_layer_names
    trainable = sum(1 for l in model.layers if l.trainable)
    print(f"   Couches entrainables : {trainable}/{len(model.layers)}")


def unfreeze_all(model):
    for layer in model.layers:
        layer.trainable = True
    print(f"   Couches entrainables : {len(model.layers)}/{len(model.layers)}")


# =============================================================================
# LOSS & MÉTRIQUES
# =============================================================================

def sparse_dice_loss(y_true, y_pred, num_classes, smooth=1e-6):
    y_true_oh    = tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), num_classes), tf.float32)
    intersection = tf.reduce_sum(y_true_oh * y_pred, axis=[1, 2])
    union        = tf.reduce_sum(y_true_oh + y_pred,  axis=[1, 2])
    return 1.0 - tf.reduce_mean((2.0 * intersection + smooth) / (union + smooth))


def combined_loss(num_classes):
    ce = tf.keras.losses.SparseCategoricalCrossentropy()
    def loss_fn(y_true, y_pred):
        return ce(y_true, y_pred) + sparse_dice_loss(y_true, y_pred, num_classes)
    loss_fn.__name__ = 'combined_loss'
    return loss_fn


# =============================================================================
# BOUCLE D'ENTRAÎNEMENT
# =============================================================================

def train_epoch(model, ds, optimizer, loss_fn, num_classes):
    total_loss = 0.0
    n_batches  = 0
    iou_metric = tf.keras.metrics.MeanIoU(num_classes=num_classes)
    for images, masks in ds:
        with tf.GradientTape() as tape:
            preds = model(images, training=True)
            loss  = loss_fn(masks, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        iou_metric.update_state(masks, tf.argmax(preds, axis=-1, output_type=tf.int32))
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
        iou_metric.update_state(masks, tf.argmax(preds, axis=-1, output_type=tf.int32))
        total_loss += loss.numpy()
        n_batches  += 1
    return float(total_loss / n_batches), float(iou_metric.result())


def run_phase(phase_num, model, train_ds, val_ds, optimizer, loss_fn,
              num_classes, num_epochs, output_dir, history, timer,
              best_val_loss, save_every):
    print(f"\n{'='*70}")
    print(f"   PHASE {phase_num}")
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
        print(f"Phase {phase_num} | Epoch {epoch+1}/{num_epochs} | "
              f"Global {global_epoch+1} | {time_stats['progress_percent']:.1f}%")
        print(f"   Train Loss: {train_loss:.4f}  |  IoU: {train_iou:.4f}")
        print(f"   Val Loss:   {val_loss:.4f}  |  IoU: {val_iou:.4f}")
        print(f"   LR: {optimizer.learning_rate.numpy():.2e}  |  "
              f"Epoch: {format_time(time_stats['epoch_time'])}  |  "
              f"Restant: {format_time(time_stats['estimated_remaining'])}  |  "
              f"ETA: {time_stats['eta'].strftime('%H:%M:%S')}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(os.path.join(output_dir, "best_model.keras"))
            print(f"   Meilleur modele sauvegarde! (val_loss={val_loss:.4f})")

        if (global_epoch + 1) % save_every == 0:
            model.save(os.path.join(output_dir, f"checkpoint_epoch_{global_epoch+1}.keras"))
            print(f"   Checkpoint epoch {global_epoch+1} sauvegarde")

    return best_val_loss


# =============================================================================
# OPTIMISATION BAYÉSIENNE (OPTUNA) — mode "optimize" uniquement
# =============================================================================

def _run_optimization(train_ds, val_ds, num_classes):
    """Recherche bayésienne sur phase1_lr et phase2_lr (sans CBAM)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_half = max(1, OPTUNA_CONFIG["n_epochs_per_trial"] // 2)

    def objective(trial):
        lr1 = trial.suggest_float("phase1_lr", 1e-4, 1e-2, log=True)
        lr2 = trial.suggest_float("phase2_lr", 1e-7, 1e-4, log=True)

        model, _ = build_model_simple(CONFIG["pretrained_model_path"], num_classes)
        loss_fn  = combined_loss(num_classes)

        # Phase 1 rapide
        freeze_base(model, {'segmentation_output_multiclass'})
        opt1 = tf.keras.optimizers.Adam(lr1)
        for _ in range(n_half):
            train_epoch(model, train_ds, opt1, loss_fn, num_classes)

        # Phase 2 rapide
        unfreeze_all(model)
        opt2     = tf.keras.optimizers.Adam(lr2)
        best_val = float('inf')
        for ep in range(n_half):
            train_epoch(model, train_ds, opt2, loss_fn, num_classes)
            val_loss, _ = val_epoch(model, val_ds, loss_fn, num_classes)
            trial.report(val_loss, ep)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            best_val = min(best_val, val_loss)

        del model
        return best_val

    os.makedirs(OPTUNA_CONFIG["output_dir"], exist_ok=True)
    study = optuna.create_study(
        direction="minimize",
        study_name=OPTUNA_CONFIG["study_name"],
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
    )

    print(f"\n{'=' * 70}")
    print(f"   OPTIMISATION BAYESIENNE — {OPTUNA_CONFIG['n_trials']} essais")
    print(f"   {OPTUNA_CONFIG['n_epochs_per_trial']} epochs/essai | sampler: TPE | pruner: Median")
    print(f"{'=' * 70}\n")

    study.optimize(objective, n_trials=OPTUNA_CONFIG["n_trials"], show_progress_bar=True)

    best = study.best_trial
    print(f"\n{'=' * 70}")
    print(f"   MEILLEUR ESSAI #{best.number}  —  val_loss: {best.value:.4f}")
    print(f"{'=' * 70}")
    for k, v in best.params.items():
        print(f"   {k}: {v}")

    report = {
        "best_trial": best.number, "best_val_loss": best.value, "best_params": best.params,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
            for t in study.trials
        ],
    }
    with open(os.path.join(OPTUNA_CONFIG["output_dir"], "optuna_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        values = [t.value for t in study.trials if t.value is not None]
        axes[0].plot(values, marker='o', linewidth=1.5)
        axes[0].set_xlabel("Essai"); axes[0].set_ylabel("Val Loss")
        axes[0].set_title("Historique Optuna"); axes[0].grid(True, alpha=0.3)
        importances = optuna.importance.get_param_importances(study)
        axes[1].barh(list(importances.keys()), list(importances.values()))
        axes[1].set_xlabel("Importance"); axes[1].set_title("Importance des hyperparametres")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OPTUNA_CONFIG["output_dir"], "optuna_results.png"), dpi=150)
        plt.close()
    except Exception:
        pass

    return best.params


# =============================================================================
# BOUCLE D'ENTRAÎNEMENT PRINCIPALE (2 phases)
# =============================================================================

def run_training(model, train_ds, val_ds, num_classes, model_config):
    loss_fn = combined_loss(num_classes)
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou':  [], 'val_iou':  [],
        'lr': [], 'epoch_times': [], 'cumulative_times': [],
        'model_config': model_config,
    }
    best_val_loss = float('inf')

    # Nom des couches entraînables en phase 1 selon le mode
    phase1_trainable = {'segmentation_output_multiclass'}
    if model_config.get('mode') == 'attention':
        phase1_trainable.add('cbam_attention')
        # Les sous-couches de CBAM sont aussi entraînables via le parent
        for layer in model.layers:
            if layer.name.startswith('cbam_attention') or layer.name.startswith('channel_att') \
                    or layer.name.startswith('spatial_att'):
                phase1_trainable.add(layer.name)

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"   PHASE 1 : Encodeur + decodeur geles — entrainement de la tete")
    print(f"   Epochs: {CONFIG['phase1_epochs']} | LR: {CONFIG['phase1_lr']}")
    freeze_base(model, phase1_trainable)
    optimizer1 = tf.keras.optimizers.Adam(learning_rate=CONFIG["phase1_lr"])
    timer1     = TrainingTimer(CONFIG["phase1_epochs"])
    timer1.start_training()

    best_val_loss = run_phase(
        phase_num=1, model=model, train_ds=train_ds, val_ds=val_ds,
        optimizer=optimizer1, loss_fn=loss_fn, num_classes=num_classes,
        num_epochs=CONFIG["phase1_epochs"], output_dir=CONFIG["output_dir"],
        history=history, timer=timer1, best_val_loss=best_val_loss,
        save_every=CONFIG["save_every"],
    )

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"   PHASE 2 : Fine-tuning complet (LR faible)")
    print(f"   Epochs: {CONFIG['phase2_epochs']} | LR: {CONFIG['phase2_lr']}")
    unfreeze_all(model)
    optimizer2 = tf.keras.optimizers.Adam(learning_rate=CONFIG["phase2_lr"])
    timer2     = TrainingTimer(CONFIG["phase2_epochs"])
    timer2.start_training()

    best_val_loss = run_phase(
        phase_num=2, model=model, train_ds=train_ds, val_ds=val_ds,
        optimizer=optimizer2, loss_fn=loss_fn, num_classes=num_classes,
        num_epochs=CONFIG["phase2_epochs"], output_dir=CONFIG["output_dir"],
        history=history, timer=timer2, best_val_loss=best_val_loss,
        save_every=CONFIG["save_every"],
    )

    # ── Finalisation ─────────────────────────────────────────────────────────
    model.save(os.path.join(CONFIG["output_dir"], "final_model.keras"))

    all_epoch_times = timer1.epoch_times + timer2.epoch_times
    total_time      = sum(all_epoch_times)
    fts = {
        'total_time':               total_time,
        'total_time_formatted':     format_time(total_time),
        'avg_epoch_time':           float(np.mean(all_epoch_times)),
        'avg_epoch_time_formatted': format_time(np.mean(all_epoch_times)),
        'min_epoch_time_formatted': format_time(np.min(all_epoch_times)),
        'max_epoch_time_formatted': format_time(np.max(all_epoch_times)),
        'std_epoch_time':           float(np.std(all_epoch_times)),
        'epoch_times':              [float(t) for t in all_epoch_times],
        'start_datetime':           timer1.training_start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        'end_datetime':             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    history['time_stats'] = fts

    with open(os.path.join(CONFIG["output_dir"], "history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    # Graphiques
    epochs_range = range(1, len(history['train_loss']) + 1)
    phase1_end   = CONFIG["phase1_epochs"]
    fig, axes    = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric, title in [
        (axes[0], ('train_loss', 'val_loss'),   'Courbes de perte'),
        (axes[1], ('train_iou',  'val_iou'),    'Mean IoU par epoch'),
    ]:
        ax.plot(epochs_range, history[metric[0]], label='Train', linewidth=2)
        ax.plot(epochs_range, history[metric[1]], label='Val',   linewidth=2)
        ax.axvline(x=phase1_end + 0.5, color='gray', linestyle='--', label='Phase 1→2')
        ax.set_xlabel('Epoch'); ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    axes[2].bar(epochs_range, history['epoch_times'], color='steelblue', alpha=0.7)
    axes[2].axhline(y=fts['avg_epoch_time'], color='red', linestyle='--',
                    label=f"Moyenne: {fts['avg_epoch_time_formatted']}")
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Temps (s)')
    axes[2].set_title('Temps par epoch'); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "training_curves.png"), dpi=150)
    plt.close()

    print("\n" + "=" * 70)
    print("   ENTRAINEMENT TERMINE")
    print("=" * 70)
    print(f"   Meilleure Val Loss: {best_val_loss:.4f}")
    print(f"   Val Loss finale:    {history['val_loss'][-1]:.4f}")
    print(f"   Val IoU finale:     {history['val_iou'][-1]:.4f}")
    print(f"   Temps total:        {fts['total_time_formatted']}")
    print(f"   Fichiers:           {CONFIG['output_dir']}/")
    print("=" * 70)

    report_path = os.path.join(CONFIG["output_dir"], "training_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("RAPPORT D'ENTRAINEMENT - KERAS U-NET CADASTRAL\n" + "=" * 70 + "\n\n")
        f.write("CONFIGURATION\n" + "-" * 50 + "\n")
        for k, v in CONFIG.items():
            f.write(f"   {k}: {v}\n")
        f.write(f"\nMODE: {model_config.get('mode','?')}\n")
        f.write(f"\nSTRATEGIE TRANSFER LEARNING\n" + "-" * 50 + "\n")
        f.write(f"   Modele source: {CONFIG['pretrained_model_path']}\n")
        f.write(f"   Phase 1 ({CONFIG['phase1_epochs']} epochs): tete seule, LR={CONFIG['phase1_lr']}\n")
        f.write(f"   Phase 2 ({CONFIG['phase2_epochs']} epochs): fine-tuning, LR={CONFIG['phase2_lr']}\n")
        f.write(f"\nPERFORMANCES\n" + "-" * 50 + "\n")
        f.write(f"   Meilleure Val Loss: {best_val_loss:.4f}\n")
        f.write(f"   Val Loss finale:    {history['val_loss'][-1]:.4f}\n")
        f.write(f"   Val IoU finale:     {history['val_iou'][-1]:.4f}\n")
        f.write(f"\nTEMPS\n   Debut: {fts['start_datetime']}\n   Fin:   {fts['end_datetime']}\n")
        f.write(f"   Total: {fts['total_time_formatted']}\n")
        f.write("\nTEMPS PAR EPOCH\n" + "-" * 50 + "\n")
        for i, t in enumerate(fts['epoch_times']):
            phase = 1 if i < CONFIG["phase1_epochs"] else 2
            f.write(f"   Epoch {i+1:3d} (Ph{phase}): {format_time(t)}\n")
    print(f"   Rapport sauvegarde: {report_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Keras U-Net+MobileNetV2 - Transfer Learning Toitures Cadastrales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes disponibles:
  simple    Transfer learning standard (Phase 1 tête + Phase 2 fine-tuning)
  attention Transfer learning + CBAM Keras injecté avant la couche de sortie
  optimize  Transfer learning standard + recherche bayésienne de phase1_lr et phase2_lr (Optuna)
        """
    )
    parser.add_argument("--mode", choices=["simple", "attention", "optimize"],
                        default="simple", help="Mode d'entraînement (défaut: simple)")
    parser.add_argument("--n-trials",       type=int, default=OPTUNA_CONFIG["n_trials"])
    parser.add_argument("--n-epochs-trial", type=int, default=OPTUNA_CONFIG["n_epochs_per_trial"])
    parser.add_argument("--cbam-reduction", type=int, default=CONFIG["cbam_reduction"])
    parser.add_argument("--cbam-kernel-size", type=int, default=CONFIG["cbam_kernel_size"],
                        choices=[3, 5, 7])
    args = parser.parse_args()

    OPTUNA_CONFIG["n_trials"]           = args.n_trials
    OPTUNA_CONFIG["n_epochs_per_trial"] = args.n_epochs_trial

    print("=" * 70)
    print("   Keras U-Net+MobileNetV2 - Transfer Learning Toitures Cadastrales")
    print(f"   Mode: {args.mode.upper()}")
    print("=" * 70)

    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nDevice: {'GPU (' + str(len(gpus)) + ')' if gpus else 'CPU'}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    num_classes = len(CONFIG["classes"])

    # ── Coefficients d'augmentation (depuis ENV vars) ────────────────────────
    aug_coeffs = load_aug_coefficients(CONFIG["classes"])

    print("\nChargement du dataset...")
    dataset   = COCOSegDataset(CONFIG["images_dir"], CONFIG["annotations_file"],
                               image_size=CONFIG["image_size"])
    train_raw_ids, val_ids = dataset.split(CONFIG["train_split"])

    # ── Augmentation par classe ──────────────────────────────────────────────
    cat_id_to_name  = dataset.cat_id_to_name
    train_aug_ids   = oversample_by_class_keras(
        train_raw_ids, dataset.coco, cat_id_to_name, aug_coeffs
    )
    print_augmentation_report_keras(
        dataset.coco, train_raw_ids, train_aug_ids, cat_id_to_name, aug_coeffs
    )
    print(f"   Train: {len(train_aug_ids)} samples  ({len(set(train_aug_ids))} images uniques)")
    print(f"   Val:   {len(val_ids)} images")

    train_ds = build_tf_dataset(dataset, train_aug_ids, CONFIG["batch_size"], train=True)
    val_ds   = build_tf_dataset(dataset, val_ids,       CONFIG["batch_size"], train=False)

    if args.mode == "simple":
        print(f"\nArchitecture: U-Net+MobileNetV2 (transfer learning standard)")
        print(f"   Source: {CONFIG['pretrained_model_path']}")
        model, _     = build_model_simple(CONFIG["pretrained_model_path"], num_classes)
        model_config = {"mode": "simple"}

    elif args.mode == "attention":
        cbam_r = args.cbam_reduction
        cbam_k = args.cbam_kernel_size
        print(f"\nArchitecture: U-Net+MobileNetV2 + CBAM Keras")
        print(f"   Source: {CONFIG['pretrained_model_path']}")
        print(f"   cbam_reduction={cbam_r}, cbam_kernel_size={cbam_k}")
        model, _     = build_model_attention(CONFIG["pretrained_model_path"], num_classes,
                                             cbam_r, cbam_k)
        model_config = {"mode": "attention", "cbam_reduction": cbam_r, "cbam_kernel_size": cbam_k}

    else:  # optimize
        print(f"\nArchitecture: U-Net+MobileNetV2 (transfer learning standard)")
        print("Lancement de l'optimisation bayesienne de phase1_lr et phase2_lr...")
        best_params = _run_optimization(train_ds, val_ds, num_classes)
        if "phase1_lr" in best_params:
            CONFIG["phase1_lr"] = best_params["phase1_lr"]
        if "phase2_lr" in best_params:
            CONFIG["phase2_lr"] = best_params["phase2_lr"]
        print(f"\nHyperparametres optimises: phase1_lr={CONFIG['phase1_lr']:.2e}, "
              f"phase2_lr={CONFIG['phase2_lr']:.2e}")
        model, _     = build_model_simple(CONFIG["pretrained_model_path"], num_classes)
        model_config = {"mode": "optimize", "best_params": best_params}

    print(f"   Architecture: {model.name}")
    print(f"   Parametres totaux: {model.count_params():,}")
    print(f"   Classes: {CONFIG['classes']}")

    run_training(model, train_ds, val_ds, num_classes, model_config)


if __name__ == "__main__":
    main()
