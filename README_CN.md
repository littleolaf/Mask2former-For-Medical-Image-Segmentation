以下内容是唐武阳用mask2former在医学影像数据集上训练自己模型的项目记录。具体学习记录可以参考我的Blog笔记：[Mask2former 学习记录](https://littleolaf.github.io/posts/learn/mask2former%E5%AD%A6%E4%B9%A0%E8%AE%B0%E5%BD%95/)  
# Mask2former环境配置
原项目Github地址：[Mask2Former](https://github.com/facebookresearch/Mask2Former)  
---
安装的软件包：
python=3.8  
pytorch\==1.9.0 torchvision\==0.10.0 cudatoolkit=11.1  
Detectron2：[Detectron2安装说明](https://detectron2.readthedocs.io/tutorials/install.html)  
&emsp;建议从源码安装：
```bash
# 在工作目录下使用下列指令下载detectron2
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```
opencv=4.8.1  
其他必备的库（在requirements.txt中）
	`pip install -r requirements.txt`
#### 全流程实例：

```bash
conda create --name mask2former python=3.8 -y
conda activate mask2former
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python

# under your working directory
git clone https://github.com/facebookresearch/detectron2.git
# 转到Mask2Former的路径下
# cd ..
git clone git@github.com:facebookresearch/Mask2Former.git
cd Mask2Former
pip install -r requirements.txt  # 安装其他必备库
cd mask2former/modeling/pixel_decoder/ops  # 为 MSDeformAttn 编译 CUDA 内核
sh make.sh
```

# 医学数据集注册与使用
## kvasir-SEG数据集介绍
[Kvasir-SEG数据集官网](https://datasets.simula.no/kvasir-seg/)  
它是胃肠道息肉图像和相应分割mask的开放获取数据集，由经验丰富的胃肠病学家手动注释和验证。
## Detectron2数据集注册
请参考Detectron2官方文档：[数据集注册](https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset)  
具体配置信息可在如下位置找到：
`mask2former/data/datasets/register_kvasir_seg_semantic.py`  
预期数据集结构如下所示：
```shell
$DETECTRON2_DATASETS/
  kvasir-SEG/
    train/
      images/
      masks/
    val/
      images/
      masks/
```
其中`$DETECTRON2_DATASETS`为环境变量中指定的内置数据集的位置。

## 模型必要配置
具体配置信息可在如下位置找到：
`configs/kvasir_seg/`  

## 与mask2former模型输入相匹配
由于kvasir dataset的mask包含三通道，因此需要在semantic mapper与sem_seg_evaluation中修改mask读入方式为灰度图。 
```python
"""修改读取mask文件方式转为灰度图读取"""
# sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double") #这个读出来是个二维数组
# 以下是为了kvasir进行修改读取mask文件方式转为灰度图读取的部分   
gt_image = Image.open(dataset_dict.pop("sem_seg_file_name")).convert('L')
gt_binary_array = np.asarray(gt_image)
sem_seg_gt = np.where(gt_binary_array > 0, 1, 0).astype("double")  # .astype(np.uint8)
```

# 项目运行与调试
运行参数与配置信息位于`./vscode/launch.json`文件中。  
&emsp;训练模型请使用`kvasir`  
&emsp;测试模型请使用`kvasir_test`  
&emsp;评估模型请使用`kvasir_evaluation`  
前端页面请使用：`streamlit run website/Interface.py` //**有待完善**  
TODO:
- [ ] 完善实验数据记录
- [x] 补充数据集格式和方法
- [ ] 模型改进创新记录