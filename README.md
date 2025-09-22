# Image-Adaptive-3DLUT
Nhật ký cài đặt và chạy thử [Image-Adaptive-3DLUT](github.com/HuiZeng/Image-Adaptive-3DLUT)

## Yêu cầu
- Sử dụng hệ điều hành Linux
- Cài Miniconda để quản lý các package và môi trường ảo

## Bước 1: Cài đặt Python và các package
Tài liệu yêu cầu sử dụng `Python 3` và cài đặt các package bên dưới.
```
numpy==1.19.2
PIL==6.1.0
torch==0.4.1
torchvision==0.2.2
opencv-python==3.4.3
```
Một số vấn đề:
- Phải chọn phiên bản `Python` phù hợp => chọn phiên bản `3.6`
- `PIL` không còn hỗ trợ => cài `pillow` thay thế
- `torch 0.4.1` không còn hỗ trợ => cài `torch 1.0.0` thay thế
- `opencv-python 3.4.3` không tìm thấy => cài `opencv-python 3.4.3.18`

### Tạo và kích hoạt môi trường
```
conda create -n envpy36 python=3.6
conda activate envpy36
```
### Cài đặt các package
```
pip install numpy==1.19.2
pip install pillow==6.1.0
pip install torch==1.0.0
pip install torchvision==0.2.2
pip install opencv-python==3.4.3.18
```
### Các lỗi gặp phải khi chưa sửa
```
ERROR: Could not find a version that satisfies the requirement PIL==6.1.0 (from versions: none)
ERROR: No matching distribution found for PIL==6.1.0
```
```
ERROR: Could not find a version that satisfies the requirement torch==0.4.1 (from versions: 1.0.0, 1.0.1, 1.1.0, 1.2.0, 1.3.0, 1.3.1, 1.4.0, 1.5.0, 1.5.1, 1.6.0, 1.7.0, 1.7.1, 1.8.0, 1.8.1, 1.9.0, 1.9.1, 1.10.0, 1.10.1, 1.10.2)
ERROR: No matching distribution found for torch==0.4.1
```
```
ERROR: Could not find a version that satisfies the requirement opencv-python==3.4.3 (from versions: 3.4.0.12, 3.4.0.14, 3.4.1.15, 3.4.2.16, 3.4.2.17, 3.4.3.18, 3.4.4.19, 3.4.5.20, 3.4.6.27, 3.4.7.28, 3.4.8.29, 3.4.9.31, 3.4.9.33, 3.4.10.35, 3.4.10.37, 3.4.11.39, 3.4.11.41, 3.4.11.43, 3.4.11.45, 3.4.13.47, 3.4.14.51, 3.4.14.53, 3.4.15.55, 3.4.16.57, 3.4.16.59, 3.4.17.61, 3.4.17.63, 3.4.18.65, 4.0.0.21, 4.0.1.23, 4.0.1.24, 4.1.0.25, 4.1.1.26, 4.1.2.30, 4.2.0.32, 4.2.0.34, 4.3.0.36, 4.3.0.38, 4.4.0.40, 4.4.0.42, 4.4.0.44, 4.4.0.46, 4.5.1.48, 4.5.2.52, 4.5.2.54, 4.5.3.56, 4.5.4.58, 4.5.4.60, 4.5.5.62, 4.5.5.64, 4.6.0.66, 4.7.0.68, 4.7.0.72, 4.8.0.74, 4.8.0.76, 4.8.1.78, 4.9.0.80, 4.10.0.82, 4.10.0.84, 4.11.0.86, 4.12.0.88)
ERROR: No matching distribution found for opencv-python==3.4.3
```

## Bước 2: Compile custom CUDA/C++ extensions for Pytorch
Tài liệu hướng dẫn biên dịnh cho 2 trường hợp tùy phiên bản `pytorch` sử dụng.

Cho `pythorch 1.x` cần làm
```
cd trilinear_cpp
sh setup.sh
```
Sửa các dòng trong file `demo_eval.py`
```
from models import * ==> from models_x import *
result = trilinear_(LUT, img) ==> _, result = trilinear_(LUT, img)
```

```
python demo_eval.py
/home/pkdat/Image-Adaptive-3DLUT/models_x.py:153: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  self.LUT = nn.Parameter(torch.tensor(self.LUT))
/home/pkdat/miniconda3/envs/envpy36/lib/python3.6/site-packages/torch/nn/modules/upsampling.py:129: UserWarning: nn.Upsample is deprecated. Use nn.functional.interpolate instead.
  warnings.warn("nn.{} is deprecated. Use nn.functional.interpolate instead.".format(self.name))
/home/pkdat/miniconda3/envs/envpy36/lib/python3.6/site-packages/torch/nn/functional.py:2423: UserWarning: Default upsampling behavior when mode=bilinear is changed to align_corners=False since 0.4.0. Please specify align_corners=True if the old behavior is desired. See the documentation of nn.Upsample for details.
  "See the documentation of nn.Upsample for details.".format(mode))
Segmentation fault (core dumped)
```

```
/home/pkdat/Image-Adaptive-3DLUT/models_x.py:153: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  self.LUT = nn.Parameter(torch.tensor(self.LUT))
/home/pkdat/miniconda3/envs/envpy36/lib/python3.6/site-packages/torch/nn/modules/upsampling.py:129: UserWarning: nn.Upsample is deprecated. Use nn.functional.interpolate instead.
  warnings.warn("nn.{} is deprecated. Use nn.functional.interpolate instead.".format(self.name))
/home/pkdat/miniconda3/envs/envpy36/lib/python3.6/site-packages/torch/nn/functional.py:2423: UserWarning: Default upsampling behavior when mode=bilinear is changed to align_corners=False since 0.4.0. Please specify align_corners=True if the old behavior is desired. See the documentation of nn.Upsample for details.
  "See the documentation of nn.Upsample for details.".format(mode))
Traceback (most recent call last):
  File "demo_eval.py", line 98, in <module>
    result = F.grid_sample(LUT, img, mode='bilinear', padding_mode='border', align_corners=True)
TypeError: grid_sample() got an unexpected keyword argument 'align_corners'
```

```
python demo_eval.py 
/home/pkdat/Image-Adaptive-3DLUT/models_x.py:153: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  self.LUT = nn.Parameter(torch.tensor(self.LUT))
/home/pkdat/miniconda3/envs/envpy36/lib/python3.6/site-packages/torch/nn/modules/upsampling.py:129: UserWarning: nn.Upsample is deprecated. Use nn.functional.interpolate instead.
  warnings.warn("nn.{} is deprecated. Use nn.functional.interpolate instead.".format(self.name))
/home/pkdat/miniconda3/envs/envpy36/lib/python3.6/site-packages/torch/nn/functional.py:2423: UserWarning: Default upsampling behavior when mode=bilinear is changed to align_corners=False since 0.4.0. Please specify align_corners=True if the old behavior is desired. See the documentation of nn.Upsample for details.
  "See the documentation of nn.Upsample for details.".format(mode))
Traceback (most recent call last):
  File "demo_eval.py", line 98, in <module>
    result = F.grid_sample(LUT, img, mode='bilinear', padding_mode='border', align_corners=True)
TypeError: grid_sample() got an unexpected keyword argument 'align_corners'
```
