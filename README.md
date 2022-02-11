# Building Information Modeling and Rendering for Artificial Intelligence (BIMrAI)

![This is an image](https://github.com/mawadhi/BIMrAI/blob/main/bimference.gif)

# Description

Teaching AI machines to detect building objects is at the foundation toward achieving AI-assisted 3D recostruction of builidings. However, there exists the challenge of acquiring enough training data for deep learning that is typically manually-annotated. This is unless if a machine can generate its own data to train itself.

This project trained artificial neural networks (ANNs) solely on realisitc computer-generated images of 3D building information models—which were parametrically and automatically-generated using the Building Information Modeling Generation Environment (BIMGenE) program developed in Grasshopper. The ANN training result demonstrated accurate semantic segmenation on arbitrary photos of buildings.

This repository documents the ANN training implementation of this project.

# Prerequisites
- Win64 Anaconda 4.8.4 + Python 3.6.10
- NVIDIA GPU + 11 GB VRAM or more
- envTensorflow: tensorflow 1.13.1, cudnn 7.6.5, cudatoolkit 10.0.130
- envPyTorch: pytorch 1.1.0, cudatoolkit 9.0

# Creating the Dataset
Files used for creating the parametric-BIM training dataset can be found [here](https://github.com/mawadhi/BIMrAI/tree/main/bimgene). They include the generated FBX files (from BIMGenE) and the HDRI images that can be used for rendering in a 3D graphics software. These same files were used to create the training data that produced the [results](https://github.com/mawadhi/BIMrAI/tree/main/results) of this project:

![This is an image](https://github.com/mawadhi/BIMrAI/blob/main/bimgene.jpg)

Renderings should inlcude object-ID render passes. Preparing images and formatting the training data depends on the ANN model. There are provided examples that can be followed. For GAN the example is found [here](https://github.com/mawadhi/BIMrAI/tree/main/tensorflow/example_training_dataset) (for Tensorflow) and [here](https://github.com/mawadhi/BIMrAI/tree/main/pytorch/datasets/example_training_dataset) (for PyTorch). For conventional semantic-segmentation CNNs, the example is found [here](https://github.com/mawadhi/BIMrAI/tree/main/segment/example_training_dataset). 
