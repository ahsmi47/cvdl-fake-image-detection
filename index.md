---
title: Evaluating the Impact of Digital Noise in AI-Generated Image Detection
---

# Evaluating the Impact of Digital Noise in AI-Generated Image Detection

## May 30, 2024

**Authors: Ahmed Ibrahim, Michal Kuchar, Krzysztof Garbowicz**

In this blog post, we present the results of our study on the impact of adding noise to AI-generated images and how it affects the detection accuracy of state-of-the-art models. This research is part of the Deep Vision Seminar Project. Our entire research, including code and datasets, can be found on our [Kaggle repository]

## 1. Introduction

The advancement of generative machine learning models has made it increasingly difficult to distinguish between real and AI-generated images. While these models offer remarkable capabilities, they also pose significant risks, such as the potential for creating highly convincing fake images that could be used maliciously. This study explores how adding digital noise to fake images can affect their detection by current state-of-the-art systems. We hypothesized that noise manipulation could deceive these systems, leading to misclassification of fake images as real.

![Example of AI-Generated Images with Noise](path/to/your/image.jpg)

## 2. Motivation

The ability to distinguish real from fake images is crucial in combating misinformation and ensuring the integrity of digital media. Existing detection methods leverage the noise patterns inherent in real images, which are absent in AI-generated images. However, if noise can be artificially added to fake images, these detection methods might be compromised. Our research aims to assess this vulnerability and provide insights for improving detection systems.

## 3. Implementation

We implemented a detection system inspired by advanced models and evaluated its performance with and without noise added to fake images. The model was trained using a diverse dataset of real and fake images, ensuring it could learn to identify the subtle differences between them.

### 3.1 Training

Our training process involved using a sophisticated deep neural network to extract features from the images. The network architecture included convolutional layers followed by fully connected layers, optimized using various algorithms such as SGD, SGD with momentum, and Adam. We employed a triplet loss function to encourage the network to learn distinguishing features between real and fake images.

![Network Architecture](path/to/your/image.jpg)

### 3.2 Deep Network

The deep network utilized for feature extraction was based on an Inception architecture, pre-trained on ImageNet. This provided a robust foundation for our model, allowing it to effectively learn and distinguish image features even in the presence of added noise.

![Deep Network Architecture](path/to/your/image.jpg)

## 4. Results

We evaluated the model's performance using accuracy metrics such as L2 norm and cosine similarity. The results demonstrated that adding noise to fake images significantly impacted the model's detection accuracy, often causing misclassification.

### 4.1 Final Result Overview

The model was trained extensively, with various optimizers used at different stages to refine its learning. Despite some challenges, including limited computing power, the model achieved promising results. The addition of noise to fake images resulted in a noticeable drop in detection accuracy, highlighting a critical weakness in current detection systems.

![Training Loss and Accuracy](path/to/your/image.jpg)

## 5. Discussion and Limitations

### 5.1 Discussion

Our research confirms that adding noise to fake images can deceive state-of-the-art detection models, leading to misclassification. This finding underscores the need for more robust detection methods that can account for noise manipulation. Future research should focus on developing adaptive detection systems that remain effective even in the presence of artificially added noise.

### 5.2 Limitations

The study faced several limitations, including the lack of implementation details from previous research, limited computational resources, and the absence of a multi-view dataset. Despite these challenges, our findings provide valuable insights into the vulnerabilities of current detection systems and suggest directions for future improvements.

## References

1. Zhu, X., et al. (2023). Generative Image Detection with Noise Patterns. *Journal of Machine Learning Research*.
2. Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering. *Google Inc*. https://doi.org/10.48550/arXiv.1503.03832

---
