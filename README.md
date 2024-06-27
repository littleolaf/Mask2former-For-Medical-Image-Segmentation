The following is a project record by Tang Wuyang for using megaformer to train my own model on medical image datasets. Specific learning records can refer to my Blog notes:[Mask2former Learning Record](https://littleolaf.github.io/posts/learn/mask2former%E5%AD%A6%E4%B9%A0%E8%AE%B0%E5%BD%95/)  
READEME in Chinese: [简体中文](README_CN.md)  
# MeGaFormer Enviroment Setup
The original Mask2Former project from Github：[Mask2Former](https://github.com/facebookresearch/Mask2Former)  
---
package to be installed: 
python=3.8  
pytorch\==1.9.0 torchvision\==0.10.0 cudatoolkit=11.1  
Detectron2：follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html)  
```bash
# install detectron2 from source in your working directory
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```
OpenCV is optional but needed by demo and visualization
opencv=4.8.1  
Other packages (in requirements.txt): 
	`pip install -r requirements.txt`
## Example conda environment setup:

```bash
conda create --name mask2former python=3.8 -y
conda activate mask2former
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python

# under your working directory
git clone https://github.com/facebookresearch/detectron2.git
git clone git@github.com:facebookresearch/Mask2Former.git
cd Mask2Former
pip install -r requirements.txt  # install other packages
cd mask2former/modeling/pixel_decoder/ops  # compile CUDA kernel for MSDeformAttn
sh make.sh
```

# Registration and use of medical datasets
Please refer to the [`./datasets/README_EN.md`](./datasets/README_EN.md) for registration and use of medical datasets.  

## Model Configuration
The specific configuration information can be found in the following location:  
`configs/`  

## Necessary modification of Detectron2 files
Please refer to the blog [synapse_for_megaformer](https://littleolaf.github.io/posts/learn/synapse_for_megaformer/)  
It records the necessary modifications to the two library files of detectron2 to run MeGaFormer correctly.  

# Operation and debugging
Please find the file `./vscode/launch.json` for running and debugging.  
&emsp;To train the model, please use:  `*` (Note : `*` represents the name of the dataset)  
&emsp;To test the model, please use: `*_test`  
&emsp;To evaluate the model, please use: `*_evaluation`  
To run the results website, please use: `streamlit run website/Interface.py` **(To be updated)**  
