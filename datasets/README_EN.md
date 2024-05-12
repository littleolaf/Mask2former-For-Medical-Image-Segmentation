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