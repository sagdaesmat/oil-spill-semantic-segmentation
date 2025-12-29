# Oil Spill Semantic Segmentation System

This project presents an end-to-end deep learning system for **pixel-level semantic segmentation of oil spill phenomena** in aerial imagery, using the **LADOS (Large-scale Aerial Dataset for Oil Spill Detection)** dataset.

The system is designed to detect and localize multiple oil-related surface patterns in complex marine environments, supporting environmental monitoring and maritime safety applications.

---

## 1. Problem Statement

Manual inspection of aerial imagery for oil spill monitoring is time-consuming and error-prone.  
This project aims to automate the process by building a deep learning–based semantic segmentation model that identifies oil-related classes at the pixel level, even in visually ambiguous scenes.

---

## 2. Dataset

- Dataset: **LADOS – Aerial Imagery Dataset for Oil Spill Detection**
- Data source: Aerial images collected from realistic marine scenarios
- Task type: **Multi-class semantic segmentation**

### Classes
1. Background  
2. Emulsion  
3. Oil  
4. Oil Platform  
5. Sheen  
6. Ship  

The dataset is characterized by:
- High intra-class variability
- Strong class imbalance
- Visual similarity between certain classes (e.g., Ship vs Oil Platform)

---

## 3. Preprocessing & Data Analysis

- Image resizing to **640 × 640** (aligned with the LADOS research paper)
- Data augmentation using Albumentations:
  - Horizontal & vertical flips
  - Rotation and scaling
  - Brightness/contrast adjustment
  - Gaussian blur
- ImageNet normalization (required for pretrained backbones)
- Pixel-wise class distribution analysis
- Class weighting and weighted sampling to mitigate class imbalance

All preprocessing, dataset analysis, and visualizations are implemented in the preprocessing notebook.

---

## 4. Model Architecture

### Encoder
- **ResNet50** pretrained on ImageNet
- Used strictly as a backbone (no ready-made segmentation models)

### Decoder
- Custom decoder with skip connections
- Progressive upsampling using bilinear interpolation

### Attention Modules (Handcrafted)
To comply with course requirements, attention modules are implemented **from scratch**:
- **Channel Attention**
- **Spatial Attention**

These modules enhance feature discrimination, especially in visually ambiguous regions.

### Output
- 1×1 convolution for pixel-wise class prediction
- Output resolution matches the input image size

---

## 5. Training Strategy

- Loss Function:
  - Cross Entropy Loss (with class weights)
  - Dice Loss
- Optimizer: AdamW
- Learning rate scheduling: ReduceLROnPlateau
- Early stopping based on validation mIoU
- Evaluation metric: **Mean Intersection over Union (mIoU)**

---

## 6. Evaluation Results

- Validation mIoU ≈ **0.54**
- Test mIoU ≈ **0.56**

Per-class IoU is reported for detailed performance analysis.

---

## 7. System Design

The system follows a modular and scalable design, separating data processing, model training, and inference.

User
│
│ (Image)
▼
Inference Notebook
│
│ Loads trained model
▼
Deep Learning Model (PyTorch)
│
│ Pixel-wise predictions
▼
Segmentation Mask + Statistics
│
▼
Visualization / Frontend Integration


### Design Principles
- Clear separation between preprocessing, training, and inference
- Offline training (not triggered by the frontend)
- Inference-ready pipeline for easy integration with a GUI or API
- Reproducible experiments via notebooks

---

## 8. Project Structure
notebooks/
├── 01_preprocessing_and_analysis.ipynb
│ - Dataset loading
│ - Augmentations
│ - Class distribution analysis
│ - Data visualization
│
├── 02_modeling_and_training.ipynb
│ - Model architecture
│ - Training loop
│ - Validation & early stopping
│ - Model checkpointing
│
└── 03_final_inference_and_visualization.ipynb
- Load trained model
- Test on unseen images
- Visualization and overlays


---

## 9. Team Contributions

This project was developed by a team of five members with clearly separated responsibilities:

- **Data Team (2 members)**
  - Dataset analysis
  - Preprocessing and augmentation
  - Class imbalance handling
  - Data visualization

- **Model Team (2 members)**
  - Model architecture design
  - Handcrafted attention modules
  - Training, evaluation, and optimization

- **Frontend Team (1 member)**
  - User interface design
  - Model integration for inference
  - Visualization of segmentation outputs

---

## 10. How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
