以下内容是唐武阳用megaformer在医学影像数据集上训练自己模型的项目记录。具体学习记录可以参考我的Blog笔记：[Mask2former 学习记录](https://littleolaf.github.io/posts/learn/mask2former%E5%AD%A6%E4%B9%A0%E8%AE%B0%E5%BD%95/)  
# MeGaFormer环境配置
Mask2Former项目Github地址：[Mask2Former](https://github.com/facebookresearch/Mask2Former)  
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
python -m pip install -e detectron2
# 转到Mask2Former的路径下
# cd ..
git clone git@github.com:facebookresearch/Mask2Former.git
cd Mask2Former
pip install -r requirements.txt  # 安装其他必备库
cd mask2former/modeling/pixel_decoder/ops  # 为 MSDeformAttn 编译 CUDA 内核
sh make.sh
```

# 医学数据集注册与使用
请参考[`./datasets/README_CN.md`](./datasets/README_CN.md)文件以获取关于注册和使用医学数据集的信息。  

## 模型必要配置
具体配置信息可在如下位置找到：
`configs/`  

## Detectron2库文件的必要修改
请参考博客[synapse_for_megaformer](https://littleolaf.github.io/posts/learn/synapse_for_megaformer/)  
其中记录了对detectron2的两个库文件的必要修改以正确运行MeGaFormer。  
# 项目运行与调试
运行参数与配置信息位于`./vscode/launch.json`文件中。  
&emsp;训练模型请使用`*` (注：`*` 代表数据集名称)  
&emsp;测试模型请使用`*_test`  
&emsp;评估模型请使用`*_evaluation`  
前端页面请使用：`streamlit run website/Interface.py` //**有待完善**  
TODO:
- [ ] 完善实验数据记录
- [x] 补充数据集格式和方法
- [ ] 模型改进创新记录