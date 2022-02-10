# Building Information Modeling and Rendering for Artificial Intelligence (BIMrAI)
Teaching AI machines to detect building objects is at the foundation toward achieving AI-assisted 3D recostruction of builidings. However, there exists the challenge of acquiring enough training data for deep learning that is typically manually-annotated. This is unless if a machine can generate its own data to train itself.

This project trained artificial neural networks (ANNs) solely on realisitc computer-generated images of 3D building information models—which were parametrically and automatically-generated using the Building Information Modeling Generation Environment (BIMGenE) program developed in Grasshopper. The ANN training result demonstrated accurate semantic segmenation on arbitrary photos of buildings.

This repository documents the ANN training implementation of this project.

# Prerequisites
- Win64 Anaconda 4.8.4 + Python 3.6.10
- NVIDIA GPU + 11 GB VRAM or more
- envTensorflow: tensorflow 1.13.1, cudnn 7.6.5, cudatoolkit 10.0.130
- envPyTorch: pytorch 1.1.0, cudatoolkit 9.0
