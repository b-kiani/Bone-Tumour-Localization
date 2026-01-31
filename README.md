# Multi-Stage Deep Learning Framework for Radiograph-Based Primary Bone Tumor Detection

This repository provides the **official implementation** of a **three-stage deep learning framework** for **detection and localization of primary bone tumors on plain radiographs**, explicitly designed to mirror the **hierarchical reasoning workflow of musculoskeletal radiologists**.

The framework integrates **large-scale musculoskeletal pretraining**, **lesion-specific fine-tuning**, and **localization-aware evaluation using FROC analysis**, with extensive robustness assessment across multiple acquisition centers.

---

##  Project Overview

Radiographic detection of primary bone tumors is inherently challenging due to:

- **Low disease prevalence**
- **High anatomical variability**
- **Subtle lesion contrast**
- **Significant inter-center acquisition heterogeneity**

To address these challenges, we propose a **decoupled, staged learning strategy** that reflects clinical reasoning while improving generalization and interpretability.

---

##  Framework Design

### **Stage-1 — Musculoskeletal Representation Pretraining**

- **Dataset:** MURA (Musculoskeletal Radiographs)  
- **Task:** Abnormal vs. normal classification  
- **Purpose:** Learn transferable MSK-specific radiographic representations  

**Architectures**
- Swin Transformer-Tiny  
- ConvNeXt-Tiny  
- EfficientNetV2-S  

**Enhancements**
- Contrast Limited Adaptive Histogram Equalization (CLAHE)  
- Gamma intensity augmentation  
- Weighted random sampling for class imbalance  

---

### **Stage-2 — Lesion-Specific Fine-Tuning**

- **Dataset:** BTXRD (Figshare Primary Bone Tumor Radiographs)  
- **Task:** Malignant vs. normal discrimination  

**Training Strategy**
- Head-only warm-up training  
- Selective backbone unfreezing  
- Low learning-rate fine-tuning  

**Evaluation Metrics**
- Classification accuracy  
- Area under the ROC curve (ROC-AUC)  

---

### **Stage-3 — Lesion Localization & Clinical Evaluation**

- **Model:** YOLOv8 object detection  

**Evaluation Protocol**
- Free-response receiver operating characteristic (FROC) analysis  
- FPPI-constrained operating points  
- Bootstrap-based confidence intervals  
- Leave-One-Center-Out (LOCO) cross-center testing  

**Error Analysis**
- Radiologist-style false-negative taxonomy for qualitative interpretation  

---

This staged design avoids end-to-end training on limited tumor datasets, reduces overfitting, and aligns model behavior with real-world musculoskeletal radiology workflows.
