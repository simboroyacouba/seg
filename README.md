# Keras-seg - Segmentation des Toitures Cadastrales

Modèle de segmentation sémantique Keras/TensorFlow pour la classification automatique des types de toitures.
**Structure identique à Mask R-CNN, DeepLabV3+ et YOLO26-seg pour comparaison équitable.**

## Structure

```
seg/
├── final_segmentation_model.keras   # Modèle entraîné
├── inference.py                     # Inférence sur nouvelles images
├── evaluate.py                      # Évaluation avec métriques COCO
├── classes.yaml                     # Configuration des classes
├── requirements.txt                 # Dépendances Python
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

Pour GPU :
```bash
pip install tensorflow[and-cuda]
```

## Utilisation

### Variables d'environnement

```bash
export SEGMENTATION_MODEL_PATH=./final_segmentation_model.keras
export SEGMENTATION_TEST_IMAGES_DIR=./test_images
export SEGMENTATION_OUTPUT_DIR=./predictions
export SEGMENTATION_DATASET_IMAGES_DIR=/chemin/vers/dataset/images
export SEGMENTATION_DATASET_ANNOTATIONS_FILE=/chemin/vers/annotations.json
export CLASSES_FILE=./classes.yaml
```

### Inférence

```bash
python inference.py
```

### Évaluation

```bash
python evaluate.py
```

## Métriques (identiques aux autres modèles)

| Métrique | Description |
|----------|-------------|
| mAP@50 | Mean Average Precision à IoU=0.5 |
| mAP@50:95 | Moyenne des AP de 0.5 à 0.95 |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1-Score | 2 × (P × R) / (P + R) |
| IoU moyen | Intersection over Union |

## Fichiers générés

### Inférence (predictions/)
```
predictions/
├── <image>_pred.png       # Visualisation
├── reports.json           # Rapports par image
├── summary.json           # Résumé global
└── summary.txt            # Résumé lisible
```

### Évaluation (evaluation/)
```
evaluation/
├── metrics.json
├── evaluation_report.txt
├── metrics_per_class.png
└── metrics_vs_iou.png
```
