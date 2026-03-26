# Adversarial Attack Project

## Requirements
pip install -r requirements.txt

## Run
python test.py

## Description
This project implements FGSM and PGD attacks on MNIST and CIFAR-10 datasets.

Results are saved in the results/ folder.

## Model

I used a pretrained ResNet18 model from PyTorch torchvision.

Reference:
https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html


## GPU (Optional)

This project runs on CPU by default.

If you want to use GPU, install PyTorch with CUDA (Python 3.10 or 3.11 recommended):

pip uninstall torch torchvision torchaudio -y

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


Then verify:

python -c "import torch; print(torch.cuda.is_available())"