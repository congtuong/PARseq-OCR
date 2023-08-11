<h1 align="center"> DPText-DETR with PARseq </h1> 

This is the repo for using "DPText-DETR: Towards Better Scene Text Detection with Dynamic Points in Transformer" with "PARseq: Scene Text Recognition with
Permuted Autoregressive Sequence Models"
---
### Introduction
Optical character recognition (OCR) is sometimes referred to as text recognition. An OCR program extracts and repurposes data from scanned documents, camera images and image-only pdfs. OCR software singles out letters on the image, puts them into words and then puts the words into sentences, thus enabling access to and editing of the original content. It also eliminates the need for manual data entry.
This repo provide the source code for using 2 SoTA, the text detector [DPText-DETR](https://github.com/ymy-k/DPText-DETR) and the scene text recognition [PARSeq](https://github.com/baudm/parseq) for this task.
### Installing
We are using ```Python3.10``` and  ```torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1```
Run this to install requirements:
```
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python scipy timm shapely albumentations Polygon3 pyproject-toml numpy==1.23.0
pip install git+https://github.com/facebookresearch/detectron2.git
pip install --upgrade setuptools
python setup.py build
pip install -e .
pip install -r requirements/core.txt -e .[train,test]
```
### Inference
Run this for inference on a image:
```
python demo/demo.py --config-file <path to config> --input <path to input image> --output <path to visualize image> --opts MODEL.WEIGHTS <path to DPText pretrain>
```
### Links
https://github.com/ymy-k/DPText-DETR
https://github.com/baudm/parseq



