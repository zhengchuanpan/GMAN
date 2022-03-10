# GMAN: A Graph Multi-Attention Network for Traffic Prediction (AAAI-2020)

<p align="center">
  <img width="600" height="450" src=./figure/GMAN.png>
</p>

This is the implementation of Graph Multi-Attention Network in the following paper: \
Chuanpan Zheng, Xiaoliang Fan*, Cheng Wang, and Jianzhong Qi. "[GMAN: A Graph Multi-Attention Network for Traffic Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/5477)", Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI-20), 2020, 34(01): 1234-1241.

## Data
The datasets are available at [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g), provided by [DCRNN](https://github.com/liyaguang/DCRNN), and should be put into the corresponding `data/` folder.

## Requirements
Python 3.7.10, tensorflow 1.14.0, numpy 1.16.4, pandas 0.24.2

## Results
<p align="center">
  <img width="900" height="400" src=./figure/results.png>
</p>

## Third-party re-implementations
A Pytorch implementaion by [VincLee8188](https://github.com/VincLee8188) is available at [GMAN-Pytorch](https://github.com/VincLee8188/GMAN-PyTorch).

## Citation

If you find this repository useful in your research, please cite the following paper:
```
@inproceedings{ GMAN-AAAI2020,
  author     = "Chuanpan Zheng and Xiaoliang Fan and Cheng Wang and Jianzhong Qi"
  title      = "GMAN: A Graph Multi-Attention Network for Traffic Prediction",
  booktitle  = "AAAI",
  pages      = "1234--1241",
  year       = "2020"
}
```
