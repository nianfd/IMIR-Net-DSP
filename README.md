# IMIR-Net
This is the official implementation (based on PyTorch) of our paper "Ingredient-guided Multi-modal Interaction and Refinement Network for RGB-D Food Nutrition Assessment" (Digital Signal Processing).
Including code, models, and log file.

<div align="center">
  <img src="https://github.com/nianfd/IMIR-Net-DSP/blob/main/framework.png" width=800 />
</div>

## Data and model
- [data with dish grounding](https://pan.baidu.com/s/1VzYF3wWGLrYJHH4liUEhDw?pwd=14x6)
- [pre-trained model on Nutrition5k dataset](https://pan.baidu.com/s/1x-TSXCddzoZjALfB0rIamQ?pwd=hz1e)
- [Ingredient feature extracted from CLIP text encoder](https://pan.baidu.com/s/159EDLOwMtHj549Cjube0YA?pwd=7f4d)
- fine-tune OFA dish lists (Nutrition5k).txt: dish lists for fine-tune OFA visual grounding mdoel.
- [food2k_resnet101_0.0001.pth](https://pan.baidu.com/s/1NEYUFdZbnku3wSReIPAqJw?pwd=j52z)


## Train
python train_rgbd.py

## Inference
python test_rgbd.py

## Result
<div align="center">
  <img src="https://github.com/nianfd/IMIR-Net-DSP/blob/main/results.png" width=800 />
</div>
