# Multi-label-Image-Classification

I have implemented a multi-label image classifier on the PASCAL VOC 2007 dataset by designing and 
training deep convolutional network to predict a binary present/absent image-level label for each of the 20 PASCAL classes. 

- **Part 1A - Pre-defined Models**:
    - Train [AlexNet](https://pytorch.org/hub/pytorch_vision_alexnet/) from scratch. ([AlexNet](https://pytorch.org/hub/pytorch_vision_alexnet/) - PyTorch built-in)
    - Fine-tune [AlexNet](https://pytorch.org/hub/pytorch_vision_alexnet/)  which is pretrained on [ImageNet](http://www.image-net.org/).
    - Train a simple network (defined in `classifier.py`) from scratch.
- **Part 1B - Self-designed Models**:
    - Used the concepts or ideas from existing models (e.g. VGG, ResNet, DenseNet, etc.)
