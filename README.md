# SimCLR-based Representation Learning on CIFAR-10

This project demonstrates the use of SimCLR (Simple Contrastive Learning of Representations) for learning image embeddings on the CIFAR-10 dataset. The learned embeddings are then used with a shallow classifier for image classification. 

## Overview

SimCLR is a self-supervised learning framework that learns representations by maximizing agreement between differently augmented views of the same image. In this project, we apply SimCLR to the CIFAR-10 dataset to learn meaningful image embeddings. The framework uses random cropping and color perturbations as augmentations, enabling the model to learn contrastive features without the need for labeled data.

After learning the representations, the encoder is combined with a shallow classifier (with just one fully connected layer) to perform image classification tasks on the CIFAR-10 dataset.

## Key Components

1. **SimCLR Representation Learning**: The model learns embeddings by applying contrastive learning techniques, using random crop and color perturbations for augmentations.
   
2. **Shallow Classifier**: A simple, one-layer fully connected classifier is used to classify images based on the learned embeddings. This demonstrates that powerful representations can be learned even with minimal downstream classifiers.

3. **CIFAR-10 Dataset**: The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## SimCLR: History and Background

SimCLR was introduced by researchers at Google Research in the paper titled *"A Simple Framework for Contrastive Learning of Visual Representations"* (2020). The main idea behind SimCLR is to learn visual representations through a contrastive loss, encouraging similar images to have similar representations while dissimilar images should be far apart in the feature space.

SimCLR uses data augmentations like random cropping, flipping, and color distortions to generate two different views (augmentations) of the same image. A neural network encoder then learns to map these augmented views of the same image to similar embeddings, while pushing embeddings of different images further apart.

The SimCLR framework relies on the following key components:

- **Data Augmentation**: Generating positive pairs by applying various transformations (such as color jitter, random cropping, etc.) to the input images.
- **Contrastive Loss**: A loss function that compares pairs of images in the embedding space and encourages the model to pull together embeddings of augmented versions of the same image while pushing apart those of different images.
- **Encoder Network**: A deep neural network (typically a ResNet-based architecture) that generates embeddings for each image.

## Model Architecture

1. **Encoder Network**: A ResNet-based encoder learns the representations of images. This model is trained using SimCLR’s contrastive loss function.
   
2. **Shallow Classifier**: After training the encoder, we use the learned embeddings with a shallow fully connected classifier (one FC layer) for classifying images into the 10 CIFAR-10 categories.

## Training Procedure

- **Augmentation**: The training procedure involves applying random crop and color perturbations to the images to generate positive pairs.
  
- **Loss Function**: The contrastive loss (NT-Xent loss) is used to train the model to bring augmented views of the same image closer together in the feature space.

- **Optimizer**: Adam optimizer is used to optimize the model's parameters during training.

## Results

The model was trained on the CIFAR-10 dataset and the learned representations were evaluated with a shallow fully connected classifier. The embeddings learned by SimCLR were found to be effective for the image classification task.

