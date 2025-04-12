# Cifar-10-Image-Classification
手工搭建三层神经网络分类器，在数据集 CIFAR-10 上进行训练以实现图像分类。

## 数据集
CIFAR-10数据集下载：http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
下载并解压，将以下文件：
`batches.meta
data_batch_1 ~ data_batch_5
test_batch`
添加至cifar-10-batches-py文件夹中


## 有关资源下载：

训练后模型权重下载：https://pan.baidu.com/s/1SYsmFFKPSP3ntSYgy4YbTA?pwd=w2tw

数据集下载：http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz



## Requirements
`numpy~=2.0.1
matplotlib~=3.9.2
tqdm~=4.67.1`

## 模型训练

进入 train.py 修改以下部分（可选）：

MLP神经网络结构参数：
`# 在这里指定每个线性层的输入维度、输出维度和后接的激活函数
nn_architecture = [
    {"input_dim": 3072, "output_dim": 128, "activation": "relu"},
    {"input_dim": 128, "output_dim": 32, "activation": "relu"},
    {"input_dim": 32, "output_dim": 10, "activation": "softmax"},
]`

数据加载器参数：
`# 在这里指定数据集所在路径、验证集大小、批量大小
dataloader_kwargs = {
    "path_dir": "cifar-10-batches-py",
    "n_valid": 2000,  
    "batch_size": 16,
}`

SGD优化器参数：
`# 在这里指定学习率、L2正则项系数、学习率衰减系数、学习率衰减步数
optimizer_kwargs = {
    "lr": 0.05,
    "ld": 0.001,
    "decay_rate": 0.95,
    "decay_step": 6000,
}`

训练器参数：
`# 在这里指定训练轮次、验证步数
trainer_kwargs = {
    "n_epochs": 100,
    "eval_step": 10,
}`

进入仓库根目录，运行：
`python train.py`

## 模型测试
将模型权重文件（一定要包括.pkl和.json文件）放至某一目录，例如models/

进入 test.py 修改以下部分（可选）：

数据加载器参数：
`# 在这里指定数据集所在路径、批量大小
dataloader_kwargs = {
    "path_dir": "cifar-10-batches-py",
    "batch_size": 16,
}`

模型权重文件的路径（指定.pkl的路径即可，.json文件会自动读取）：
`ckpt_path = "models/model_epoch_75.pkl"`

进入仓库根目录，运行：
`python test.py`

## 超参数组合

在 gridsearch.py 中设置部分超参数的默认值和选项表（超参数名要严格与nn_architecture对应)
随后设置网格搜索超参数组合的基准（loss或者acc）以及其他参数

进入仓库根目录，运行：
`python gridsearch.py`

超参数组合的搜索结果会自动保存在：
`gridsearch_results.json`

## 训练参数可视化

.\visualization/ParamVis.py提供了对模型网络初始化和训练后各层参数的可视化代码（包括直方图和热力图）

进入仓库根目录，运行：
`python .\visualization/ParamVis.py`

可视化结果保存在：
`.\visualization\images`

 