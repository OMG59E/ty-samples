# TySamples

基于dcl接口用python/C++实现的常见模型推理示例，目前主要包括检测、分类模型。

## 适用设备

- SOM开发板(A300)

## 目录结构

```
modelzoo
├── examples      c++模型示例
├── python        python模型示例
├── src           源码目录
│   ├── models    模型推理实现模块
│   └── utils     常用函数
├── data          demo数据
├── build.sh      编译脚本
├── CMakeLists.txt
└── README.md
```

## 支持模型
- yolov3
- yolov5 (det, seg)
- yolov6
- yolov7 (det, pose)
- yolov8 (det, pose, seg)
- yolo-nas
- mobilenet_v1
- mobilenet_v2
- resnet18
- resnet50
- PPLCNet_1x1
- retinaface_mobilenet_v1_x0.25

## 使用说明

### 编译
```bash
# 安装依赖
apt update
apt install cmake gcc g++ make git nfs-kernel-server python3.8-dev
#
git clone https://github.com/OMG59E/ty-samples.git
cd ty-samples
cd 3rdparty
# 安装python依赖
python3 get-pip.py -i https://mirror.baidu.com/pypi/simple
pip3 install tqdm opencv-python==4.5.4.60 numpy==1.24.4 pycocotools==2.0.7 -i https://mirror.baidu.com/pypi/simple
# 编译opencv
sh get_opencv.sh
sh build_opencv.sh
# 
cd .. && sh build.sh
```

### 执行

- 检测模型，以yolov7为例

```bash
# c++
cd build
./dcl_infer_yolov7 /config/sdk.cfg ../data/test.bmp /DEngine/tymodelzoo/detection/onnx_yolov7/dp2000/net_combine.ty ../data/test_res.bmp
# python
```

- 分类模型，以resnet为例

```bash
# c++
cd build
./dcl_infer_resnet /config/sdk.cfg ../data/test.bmp /DEngine/tymodelzoo/classification/caffe_resnet50/dp2000/net_combine.ty
# python

```