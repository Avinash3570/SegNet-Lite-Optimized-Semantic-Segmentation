# ﻿**SegNet: Semantic Segmentation with VGG and EfficientNet-B0**

# **🚀 Project Overview**

This project implements the **SegNet** architecture for semantic segmentation, following the guidelines from the original paper ([SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/pdf/1511.00561.pdf)). We experiment with two backbone models:

1. **Original SegNet with VGG-16** 🔥
1. **Lightweight SegNet with EfficientNet-B0** ⚡

We also analyze the trade-offs in accuracy and efficiency between these models.

# **📂 Dataset:**
Pascal VOC

We use the [Pascal VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/) for training and evaluation, a benchmark dataset for object recognition and segmentation.

# **✅ Data Preprocessing & Loading**

- **Download & Preprocess** the Pascal VOC dataset
- **Set up Data Loaders** for training and validation
- **Apply Augmentations** to improve model generalization

# **🏗️ Model Architecture**

**🔹 Original SegNet (VGG-16 Backbone)**

- Fully Convolutional Encoder-Decoder network
- Uses VGG-16-based encoder
- Decoder with max unpooling layers for upsampling

**🔹 Lightweight SegNet (EfficientNet-B0 Backbone)**

- Replaces VGG-16 with EfficientNet-B0 for improved efficiency
- Reduces computational complexity while maintaining performance

# **🏋️ Training & Evaluation**

- **Monitor Training Progress** using metrics such as:
  - Loss
  - Pixel Accuracy
- **Evaluate Model Performance** on the test dataset
  - Pixel Accuracy
  - Mean IoU (Intersection over Union)
  - Dice Score (F1 Score for segmentation)
  - Jaccard Index (IoU)

# **🔧 Hyperparameter Tuning**

We fine-tune hyperparameters like:

- Learning Rate
- Batch Size
- Optimizer (Adam, SGD)
- Number of Epochs

# **📜 References**

- [SegNet Paper](https://arxiv.org/pdf/1511.00561.pdf)
- [Pascal VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)

  
![2](https://github.com/user-attachments/assets/0ae6c60d-e89c-4487-9172-563ac7f51b36)

![1](https://github.com/user-attachments/assets/3bd391ba-6aef-45cf-9a71-53ac16593a1d)

