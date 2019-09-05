# Face Classification and Verification
This repository contains code to implement image classification and verification using a ResNet50 network.

# Machine Learning Library
The code is written in PyTorch.

# Essentials
The code is contained within the `tools\` directory. There are five modules, whose functions are as follows:
- `test_vrf.py`: Takes two images as input, then outputs the cosine similarity between the both.
- `test_cls.py`: Takes an image as input, then outputs the label.
- `runner.py`: Trains the network.
- `model.py`: ResNet50 network.
- `dataset.py`: Implements the dataset class for both image classification and verification.
