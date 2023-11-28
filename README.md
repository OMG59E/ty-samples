# MODELZOO

基于dcl接口实现的常见模型推理示例，目前主要包括检测、分类模型。

## 目录结构

```
modelzoo
├── examples      c++模型示例
├── python
├── src           源码目录
│   ├── codecs    jpeg编解码模块(暂不能用)
│   ├── models    模型推理实现模块
│   └── utils     常用函数
├── data          demo数据
├── build.sh      交叉编译脚本，用于芯片运行
├── build_simu.sh 仿真编译脚本，用于软仿(不建议使用，巨慢)
├── CMakeLists.txt
└── README.md
```

## 使用说明

```shell
apt update
apt install cmake gcc g++ make python3.8-dev
pip3 install opencv-python==4.5.2.54 numpy
```

### 执行

在芯片端执行，需要自行挂载主机侧/DEngine至芯片侧

#### 检测模型

以yolov7为例, 对应模型的芯片模型需要在tymodelzoo中自行编译

```shell
cd /DEngine/tyexamples/modelzoo/build/
./dcl_infer_yolov7 /config/sdk.cfg ../data/test.bmp /DEngine/tymodelzoo/detection/onnx_yolov7/dp2000/net_combine.bin ../data/test_res.bmp
```

#### 分类模型

以resnet为例, 对应模型的芯片模型需要在tymodelzoo中自行编译

```shell
cd /DEngine/tyexamples/dcl/modelzoo/build/
./dcl_infer_resnet /config/sdk.cfg ../data/test.bmp /DEngine/tymodelzoo/classification/caffe_resnet50/dp2000/net_combine.bin
```