# 为MeGaFormer准备数据集
在MeGaFormer中，我们使用Detectron2库来加载数据集数据。可以通过访问 `DatasetCatalog` 来获取数据集的数据，或者通过访问 `MetadataCatalog` 来获取其元数据（类名等）。本文档介绍如何正确设置数据集格式以便于上述API可以使用他们。  

## 预期数据集格式：
请参考Detectron2官方文档：[数据集注册](https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset)  
### [kvasir-SEG数据集](https://datasets.simula.no/kvasir-seg/)
它是胃肠道息肉图像和相应分割mask的开放获取数据集，由经验丰富的胃肠病学家手动注释和验证。  
具体配置信息可在如下位置找到： `mask2former/data/datasets/register_kvasir_seg_semantic.py`
预期数据集结构如下所示：  
```bash
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
### DRIVE数据集
具体配置信息可在如下位置找到： `mask2former/data/datasets/register_drive_semantic_seg.py`  
预期数据集结构如下所示：  
```shell
$DETECTRON2_DATASETS/
  DRIVE/
    training/
      images/
      1st_manual/
    test/
      images/
      1st_manual/
```  
### Synapse数据集
具体配置信息可在如下位置找到： `mask2former/data/datasets/register_drive_semantic.py`  
预期数据集结构如下所示：  
```bash
$DETECTRON2_DATASETS/
  Synapse/
    
```  