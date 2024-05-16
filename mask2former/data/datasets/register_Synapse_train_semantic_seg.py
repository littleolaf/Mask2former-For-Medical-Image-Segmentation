from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import os

# 'blood vessel' is the only category in DRIVE dataset
SYNAPSE_TRAIN_CATEGORIES = [
                        #  {"name": "other", "id": 0, "trainId": 0,}, # 'color': [0, 0, 0],                         
                        #  {"name": "aorta主动脉", "id": 1, "trainId": 1,}, # 对
                        #  {"name": "gallbladder胆囊", "id": 2, "trainId": 2,}, # 对
                        #  {"name": "left kidney左肾脏", "id": 3, "trainId": 3,}, # 对
                        #  {"name": "right kidney右肾脏", "id": 4, "trainId": 4,}, # 对
                        #  {"name": "esophagus食管", "id": 5, "trainId": 5,}, # not in test
                        #  {"name": "liver肝", "id": 6, "trainId": 6,}, # 对
                        #  {"name": "pancreas胰腺", "id": 7, "trainId": 7,},
                        #  {"name": "spleen脾", "id": 8, "trainId": 8,},# 对
                        #  {"name": "inferior vena cava下腹静脉", "id": 9, "trainId": 9,}, # not in test
                        #  {"name": "portal & splenic vein门/脾 静脉", "id": 10, "trainId": 10,}, # not in test
                        #  {"name": "stomach胃", "id": 11, "trainId": 11,}, # 对
                        #  {"name": "right adrenal gland右肾上腺", "id": 12, "trainId": 12,}, # not in test
                        #  {"name": "left adrenal gland左肾上腺", "id": 13, "trainId": 13,}, # not in test
                         {"name": "", "id": 0, "trainId": 0,}, # 'color': [0, 0, 0],                         
                         {"name": "", "id": 1, "trainId": 1,}, # 对
                         {"name": "", "id": 2, "trainId": 2,}, # 对
                         {"name": "", "id": 3, "trainId": 3,}, # 对
                         {"name": "", "id": 4, "trainId": 4,}, # 对
                         {"name": "", "id": 5, "trainId": 5,}, # not in test
                         {"name": "", "id": 6, "trainId": 6,}, # 对
                         {"name": "", "id": 7, "trainId": 7,},
                         {"name": "", "id": 8, "trainId": 8,},# 对
                         {"name": "", "id": 9, "trainId": 9,}, # not in test
                         {"name": "", "id": 10, "trainId": 10,}, # not in test
                         {"name": "", "id": 11, "trainId": 11,}, # 对
                         {"name": "", "id": 12, "trainId": 12,}, # not in test
                         {"name": "", "id": 13, "trainId": 13,}, # not in test
                        ] #

def get_Synapse_meta():
    stuff_ids = [k["id"] for k in SYNAPSE_TRAIN_CATEGORIES]
    assert len(stuff_ids) == 14, len(stuff_ids)

    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in SYNAPSE_TRAIN_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes
    }
    return ret


def register_Synapse(name, root):
    dataset_name = f"Synapse_{name}"
    image_dir = os.path.join(root, "images")
    gt_dir = os.path.join(root, "labels")
    # 注册数据集
    DatasetCatalog.register(
        dataset_name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="npz", image_ext="npz")
    )
    # 设置元数据
    MetadataCatalog.get(dataset_name).set(

        image_root=image_dir,
        sem_seg_root=gt_dir,
        evaluator_type="sem_seg",
        ignore_label=0,
        **get_Synapse_meta(),
    )

def register_all_Synapse(root):
    # 移除已注册的数据集
    # print(DatasetCatalog.list())
    for name in ["train"]:
        key = f"Synapse_{name}"
        if key in  DatasetCatalog.list():
            print(f"Remove registered dataset: {key}")
            DatasetCatalog.remove(key)
    
    root = os.path.join(root, 'Synapse') # please change this path to your own请替换为自己的路径
    name = "train"
    ROOT = os.path.join(root, name)
    register_Synapse(name, ROOT)

_root = os.getenv("DETECTRON2_DATASETS", "/home/tangwuyang/Dataset/")  # Update this path
register_all_Synapse(_root)
