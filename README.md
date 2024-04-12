The following is a project record by Tang Wuyang for using mask2former to train my own model on medical image datasets. Specific learning records can refer to my Blog notes:  
TODO：To be updated  
READEME in Chinese: [README](README_CN.md)  
# Mask2former Enviroment Setup
The original project from Github：[Mask2Former](https://github.com/facebookresearch/Mask2Former)  
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
## Introduction of kvasir-SEG dataset
[Kvasir-SEG Website](https://datasets.simula.no/kvasir-seg/)  
It is an open-access dataset of gastrointestinal polyp images and corresponding segmentation masks, manually annotated and verified by an experienced gastroenterologist.
## Register the dataset in Detectron2
Please refer to Detectron2 's official documentation: [Register a Dataset](https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset)  

The specific configuration information can be found in the following location:
`mask2former/data/datasets/register_kvasir_seg_semantic.py`  

Expected dataset structure for kvasir-SEG: 
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
`$DETECTRON2_DATASETS` is the environment variable specify the location of the datasets.

## Model Configuration
The specific configuration information can be found in the following location:  
`configs/kvasir_seg/`  

## Different from mask2former
Since the mask of kvasir dataset contains three channels, it is necessary to modify the mask reading method to grayscale in `mask2former/data/dataset_mappers/mask_former_semantic_dataset_mapper.py` and sem_seg_evaluation in Detectron2.
### Example:
```python
"""original code"""
# sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
"""modified code""" 
gt_image = Image.open(dataset_dict.pop("sem_seg_file_name")).convert('L')
gt_binary_array = np.asarray(gt_image)
sem_seg_gt = np.where(gt_binary_array > 0, 1, 0).astype("double")
```

# Operation and debugging
Please find the file `./vscode/launch.json` for running and debugging.  
&emsp;To train the model, please use:  `kvasir`  
&emsp;To test the model, please use: `kvasir_test`  
&emsp;To evaluate the model, please use: `kvasir_evaluation`  
To run the results website, please use: `streamlit run website/Interface.py` **(To be updated)**  
