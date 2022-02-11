# Building Information Modeling and Rendering for Artificial Intelligence (BIMrAI)

![This is an image](https://github.com/mawadhi/BIMrAI/blob/main/bimference.gif)

## Description

Teaching AI machines to detect building objects is at the foundation toward achieving AI-assisted 3D reconstruction of buildings. However, there exists the challenge of acquiring enough training data for deep learning that is typically manually annotated. This is unless a machine can generate its own data to train itself.

This project trained artificial neural networks (ANNs) solely on realistic computer-generated images of 3D building information models—which were parametrically and automatically generated using the Building Information Modeling Generation Environment (BIMGenE) program developed in Grasshopper. The ANN training result demonstrated accurate semantic segmentation on arbitrary photos of buildings.

This repository documents the ANN training implementation of this project.

## Prerequisites
- Win64 Anaconda 4.8.4 + Python 3.6.10
- NVIDIA GPU + 11 GB VRAM or more
- envTensorflow: tensorflow 1.13.1, cudnn 7.6.5, cudatoolkit 10.0.130
- envPyTorch: pytorch 1.1.0, cudatoolkit 9.0

## Creating the Dataset
Files used for creating the parametric-BIM training dataset can be found [here](https://github.com/mawadhi/BIMrAI/tree/main/bimgene). They include the generated FBX files (from BIMGenE) and the HDRI images that can be used for rendering in 3D graphics software. These same files were used to create the training data that produced the [results](https://github.com/mawadhi/BIMrAI/tree/main/results) of this project.

![This is an image](https://github.com/mawadhi/BIMrAI/blob/main/bimgene.jpg) 

![This is an image](https://github.com/mawadhi/BIMrAI/blob/main/bimgene2.jpg)

Renderings should include object-ID render passes. Preparing images and formatting the training data depends on the ANN model. There are provided examples that can be followed. For image-to-image translation GAN the example is found [here](https://github.com/mawadhi/BIMrAI/tree/main/tensorflow/example_training_dataset) (for Tensorflow) and [here](https://github.com/mawadhi/BIMrAI/tree/main/pytorch/datasets/example_training_dataset) (for PyTorch). For conventional semantic-segmentation CNNs, the example is found [here](https://github.com/mawadhi/BIMrAI/tree/main/segment/example_training_dataset). 

## TensorFlow

### Image-to-Image Translation Train/Test

This ANN model is a TensorFlow implementation of Pix2pix (Isola et al. 2016) by [affinelayer](https://github.com/affinelayer/pix2pix-tensorflow). 
The original paper can be found in the following link: [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004v1).

- Navigate to the Tensorflow environment in the Anaconda prompt, then, to train the model:
```
cd tensorflow
python pix2pix.py --mode train --output_dir example_training_dataset_train --max_epochs 200 --input_dir example_training_dataset/train --which_direction AtoB
```

- To test the model:
```
cd tensorflow
python pix2pix.py --mode test --output_dir example_training_dataset_test --input_dir example_training_dataset/test --checkpoint example_training_dataset_train
```

### Semantic Segmentation Train/Test

The following models were tested: [BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1808.00897), [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861), and [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105) (implementations by [GeorgeSeif](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite)).

- To train the model:
```
cd segment
python train.py --num_epochs 200 --dataset example_training_dataset --crop_height 256 --crop_width 256 --batch_size 1 --num_val_images 1 --model BiSeNet
```

- To test the model:
```
cd segment
python test.py --dataset example_training_dataset --crop_height 256 --crop_width 256 --model BiSeNet --checkpoint_path checkpoints/latest_model_example_training_dataset.ckpt
```

To train/test using MobileUNET or PSPNet, simply replace BiSeNet with the desired model. Note: PSPNet requires a ResNet101 front end. To train add ```--frontend ResNet101``` at the end.

## PyTorch

This ANN model is a PyTorch implementation of Pix2pix (Isola et al. 2016) by [junyanz](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). 

### Image-to-Image Translation Train/Test

- Navigate to the PyTorch environment in the Anaconda prompt, then, to train the model:
```
cd pytorch
python train.py --dataroot ./datasets/example_training_dataset --name example_training_dataset_pix2pix --model pix2pix --batch_size 1 --n_epochs 25 --n_epochs_decay 25 --direction AtoB
```

- To test the model:
```
cd pytorch
python test.py --dataroot ./datasets/example_training_dataset --name example_training_dataset_pix2pix --model pix2pix --num_test 1 --epoch 50 --no_dropout --direction AtoB
```

## Evaluation

To use the evaluation tool, prepare the evaluation dataset in segment/eval_dataset, then run the following:
```
cd segment
python eval.py
```
This will create an "Evaluate" folder in the same directory. The evaluation metrics will be listed in the terminal and saved in a .csv file titled "test_scores".

## Results

The following test scores were achieved after training Pix2pix with the parametric-BIM dataset for approximately 4 days. The training dataset size was 660,000 images and the batch size was 128. The learning rate was set to 0.0002 and the model was trained for 50 epochs. The only augmentation done was random flipping of the images. The original test case was a set of 110 photos of a building sample that was used to validate the model. The arbitrary test photos were acquired from Wikimedia Commons. They include 12 hand-picked results and 12 randomly sampled ones with manually annotated ground truths.

|Test Sample|Accuracy (%)|Mean IoU|
|-|-|-|
|Original Test Case|89.64|0.517|
|Arbitrary Photos (Hand-Picked)|86.39|0.672|
|Arbitrary Photos (Random)|80.52|0.561|

The following images are a small sample of the results on the arbitrary building photos (all evaluated samples can be found in the results folder):

|Test Photo|Ground Truth|Prediction|
|-|-|-|
|<img src="https://github.com/mawadhi/BIMrAI/blob/main/results/gan_parametric_bim/eval_dataset/test/1%20(1).png" width="125" />|<img src="https://github.com/mawadhi/BIMrAI/blob/main/results/gan_parametric_bim/Evaluate/1%20(1)_gt.png" width="125" />|<img src="https://github.com/mawadhi/BIMrAI/blob/main/results/gan_parametric_bim/Evaluate/1%20(1)_pred.png" width="125" />|
|<img src="https://github.com/mawadhi/BIMrAI/blob/main/results/gan_parametric_bim/eval_dataset/test/1%20(5).png" width="125" />|<img src="https://github.com/mawadhi/BIMrAI/blob/main/results/gan_parametric_bim/Evaluate/1%20(5)_gt.png" width="125" />|<img src="https://github.com/mawadhi/BIMrAI/blob/main/results/gan_parametric_bim/Evaluate/1%20(5)_pred.png" width="125" />|
|<img src="https://github.com/mawadhi/BIMrAI/blob/main/results/gan_parametric_bim/eval_dataset/test/1%20(6).png" width="125" />|<img src="https://github.com/mawadhi/BIMrAI/blob/main/results/gan_parametric_bim/Evaluate/1%20(6)_gt.png" width="125" />|<img src="https://github.com/mawadhi/BIMrAI/blob/main/results/gan_parametric_bim/Evaluate/1%20(6)_pred.png" width="125" />|
|<img src="https://github.com/mawadhi/BIMrAI/blob/main/results/gan_parametric_bim_random/eval_dataset/test/1%20(8).png" width="125" />|<img src="https://github.com/mawadhi/BIMrAI/blob/main/results/gan_parametric_bim_random/Evaluate/1%20(8)_gt.png" width="125" />|<img src="https://github.com/mawadhi/BIMrAI/blob/main/results/gan_parametric_bim_random/Evaluate/1%20(8)_pred.png" width="125" />|
