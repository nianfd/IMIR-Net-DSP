# IMIR-Net
This is the official implementation (based on PyTorch) of our paper "Ingredient-guided Multi-modal Interaction and Refinement Network for RGB-D Food Nutrition Assessment".
Including code, models, and log file.

<div align="center">
  <img src="https://github.com/nianfd/IMIR-Net/blob/main/framework.png" width=800 />
</div>

## Data and model
- [data with dish grounding](https://pan.baidu.com/s/1VzYF3wWGLrYJHH4liUEhDw?pwd=14x6)
- [pre-trained model on Nutrition5k dataset](https://pan.baidu.com/s/1x-TSXCddzoZjALfB0rIamQ?pwd=hz1e)
- [pre-trained ResNet-101 on Food2K](https://pan.baidu.com/s/1CZ22j3WzZOoBKKWM8YuVwg?pwd=bpaf)

## Train
python train_rgbd.py

## Inference
python test_rgbd.py

## Result
<div align="center">
  <img src="https://github.com/nianfd/IMIR-Net/blob/main/results.png" width=800 />
</div>
