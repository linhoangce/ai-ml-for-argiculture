# Weed Detection with RF-DETR

This project demonstrates how to fine-tune an RF-DETR (Real-time Detection Transformer) model on a custom weed detection dataset using HuggingFace Transformers, Roboflow, and Weights & Biases (W&B).

## Table of Contents
- [Project Overview](#project-overview)
- [Setup](#setup)
- [Dataset Preparation](#dataset-preparation)
- [Model Fine-tuning](#model-fine-tuning)
- [Inference and Evaluation](#inference-and-evaluation)
- [Inference on Unseen Data](#inference-on-unseen-data)
- [Live Demo (Inference on Unseen Local Images)](#live-demo-inference-on-unseen-local-images)

## Project Overview
This notebook outlines the process of training an object detection model to identify weeds in agricultural images. We leverage the RF-DETR architecture, known for its real-time performance and accuracy. The project covers data loading from Roboflow, data preprocessing, model fine-tuning, evaluation of the fine-tuned model, and inference on both test and unseen datasets.

## Setup

### Configure API Keys
To run this notebook, you need to provide API keys for HuggingFace, Roboflow, and Weights & Biases. Please follow the instructions to store them securely in Colab's Secrets manager:

- **HuggingFace Token**: Store under the name `HF_TOKEN`.
- **Roboflow API Key**: Store under the name `ROBOFLOW_API_KEY`.
- **Weights & Biases API Key**: Store under the name `WANDB_API_KEY`.

### Install Dependencies
Key libraries required include `transformers`, `supervision`, `accelerate`, `roboflow`, `torchmetrics`, `albumentations`, and `rfdetr`.

```python
!pip install -q git+https://github.com/huggingface/transformers.git
!pip install -q git+https://github.com/roboflow/supervision.git
!pip install -q accelerate
!pip install -q roboflow
!pip install -q torchmetrics
!pip install -q albumentations
!pip install -q rfdetr
!pip install -q "rfdetr[train,loggers]" # For W&B integration
!pip install -q wandb
```

### Imports
Essential libraries are imported at the beginning for various tasks like data handling, model loading, training, and visualization.

## Inference with Pre-trained RT-DETR Model
A pre-trained RF-DETR model is demonstrated for quick inference on a sample image to show its out-of-the-box capabilities.

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium()
model.optimize_for_inference()

detections = model.predict("https://media.roboflow.com/dog.jpg", threshold=0.5)
# ... (visualization code)
```

## Dataset Preparation

### Download Dataset from Roboflow Universe
The custom weed detection dataset is downloaded from Roboflow using the provided API key. The dataset is typically in COCO format.

```python
from roboflow import Roboflow

ROBOFLOW_API_KEY = userdata.get("ROBOFLOW_API_KEY")
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("gurmehaks-workspace").project("weed-detector-v2")
version = project.version(4)
dataset = version.download("coco")
```

### Load Datasets
The downloaded COCO dataset is loaded into `supervision.DetectionDataset` objects for training, validation, and testing.

```python
import supervision as sv

ds_train = sv.DetectionDataset.from_coco(images_directory_path=f"{dataset.location}/train", annotations_path=f"{dataset.location}/train/_annotations.coco.json")
ds_valid = sv.DetectionDataset.from_coco(images_directory_path=f"{dataset.location}/valid", annotations_path=f"{dataset.location}/valid/_annotations.coco.json")
ds_test = sv.DetectionDataset.from_coco(images_directory_path=f"{dataset.location}/test", annotations_path=f"{dataset.location}/test/_annotations.coco.json")
```

### Display Dataset Sample
A grid of sample images with their annotations is displayed to visualize the dataset content.

### Preprocess the Data for Fine-tuning (HuggingFace Transformers)

An `AutoImageProcessor` is initialized for preparing images. Augmentations using `Albumentations` are defined for the training set to prevent overfitting, and annotations are reformatted to meet the RT-DETR model's expectations.

```python
from transformers import AutoImageProcessor
import albumentations as A

IMAGE_SIZE = 480
processor = AutoImageProcessor.from_pretrained(CHECKPOINT, do_resize=True, size={"width": IMAGE_SIZE, "height": IMAGE_SIZE})

train_augmentation_and_transform = A.Compose([...])
valid_transform = A.Compose([...])
```

A custom `PyTorchDetectionDataset` class is implemented to integrate `supervision.DetectionDataset` with the `transformers` processor and `Albumentations` for data loading and augmentation during training.

A `collate_fn` is defined to correctly batch the processed data for the `Trainer`.

### Preparing Function to Compute mAP
A `MAPEvaluator` class is defined to compute Mean Average Precision (mAP) and related metrics during model evaluation, integrating with `torchmetrics`.

## Model Fine-tuning

### Load Model
The `AutoModelForObjectDetection` is loaded from a pre-trained checkpoint, configuring `id2label`, `label2id`, and `ignore_mismatched_sizes=True` to adapt it to the custom dataset's class structure.

```python
from transformers import AutoModelForObjectDetection

model = AutoModelForObjectDetection.from_pretrained(
    CHECKPOINT,
    id2label=id2label,
    label2id=label2id,
    anchor_image_size=None,
    ignore_mismatched_sizes=True
)
```

### Setup Weights & Biases for Tracking
W&B is initialized to track training progress, metrics, and model artifacts.

### Training Arguments
`TrainingArguments` are configured, including output directory, number of epochs, learning rate, batch size, and logging preferences.

```python
from transformers import TrainingArguments
import os

training_args = TrainingArguments(
    output_dir=f"{dataset.name.replace(" ", "-")}-finetune",
    num_train_epochs=200,
    per_device_train_batch_size=16,
    report_to="wandb",
    run_name="rtdetr-weed-detection-run1",
    # ... other arguments
)
```

### Train the Model
The `Trainer` from HuggingFace is used to manage the fine-tuning process. It takes the model, training arguments, datasets, data collator, and the custom `compute_metrics` function.

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=pytorch_dataset_train,
    eval_dataset=pytorch_dataset_valid,
    data_collator=collate_fn,
    compute_metrics=eval_compute_metrics_fn
)

trainer.train()
```

## Inference and Evaluation

### Perform Inference on Test Dataset
The fine-tuned model performs inference on the test dataset. Predictions are collected and compared against ground truth annotations.

```python
# ... (model loading and prediction loop)

# Save COCO predictions
with open(OUTPUT_PREDICTION_PATH, "w") as f:
  json.dump(coco_output, f, indent=2)
```

### Visualize Test Predictions
Sample visualizations comparing ground truth and model predictions are generated.

### Calculate Evaluation Metrics
mAP@50, mAP@50:95, Precision, Recall, and F1-Score are computed and displayed for the test dataset. A plot of Precision/Recall/F1 vs. Confidence Threshold and a Precision-Recall curve are also generated.

## Inference on Unseen Data

### Mount Google Drive
Google Drive is mounted to access unseen images.

### Perform Inference on Unlabeled Images
The fine-tuned model is used to detect weeds in completely unseen images from a specified Google Drive folder. Predictions are saved in COCO format, and visualizations are generated for each image.

```python
UNLABELED_IMAGE_DIR = "/content/drive/MyDrive/Comp 4800/Photos"
OUTPUT_UNLABELED_PREDICTION_PATH = "/content/drive/MyDrive/Comp 4800/Predictions/unlabeled_predictions.coco.json"
OUTPUT_UNLABELED_VIZ_DIR = "/content/drive/MyDrive/Comp 4800/Predictions/unlabeled_visualizations"

model_unlabeled = RFDETRMedium(pretrain_weights=CHECKPOINT)
model_unlabeled.optimize_for_inference()

# ... (prediction loop and visualization saving)
```

### Display Sample of Unseen Predictions
A random sample of predictions on unseen data is displayed to visually inspect the model's performance on new images.

## Live Demo (Inference on Unseen Local Images)
This section provides a demonstration of running inference on unseen images directly uploaded to the Colab environment. The images are processed, predictions are made, and visualizations are saved locally.

```python
UNLABELED_IMAGE_DIR = "/content" # Use /content folder for unseen images
OUTPUT_UNLABELED_PREDICTION_PATH = "/content/unseen_predictions.coco.json"
OUTPUT_UNLABELED_VIZ_DIR = "/content/unseen_visualizations"

# ... (same inference and visualization logic as above, but with /content paths)
```
