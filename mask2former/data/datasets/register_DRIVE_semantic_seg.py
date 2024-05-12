from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import os

# 'blood vessel' is the only category in DRIVE dataset
DRIVE_CATEGORIES = [{"name": "other", "id": 0, "trainId": 0,'color': [0, 0, 0],},
                         {"name": "blood vessel", "id": 1, "trainId": 1,'color': [255, 255, 255],}
                        ] #

def get_DRIVE_meta():
    stuff_ids = [k["id"] for k in DRIVE_CATEGORIES]
    assert len(stuff_ids) == 2, len(stuff_ids)

    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in DRIVE_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes
    }
    return ret


def register_DRIVE(name, root):
    dataset_name = f"DRIVE_{name}"
    image_dir = os.path.join(root, "images")
    gt_dir = os.path.join(root, "1st_manual")
    # 注册数据集
    DatasetCatalog.register(
        dataset_name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="gif", image_ext="tif")
    )
    # 设置元数据
    MetadataCatalog.get(dataset_name).set(

        image_root=image_dir,
        sem_seg_root=gt_dir,
        evaluator_type="sem_seg",
        ignore_label=10,
        **get_DRIVE_meta(),
    )

def register_all_DRIVE(root):
    # 移除已注册的数据集
    # print(DatasetCatalog.list())
    for name in ["training", "test"]:
        key = f"DRIVE_{name}"
        if key in  DatasetCatalog.list():
            print(f"Remove registered dataset: {key}")
            DatasetCatalog.remove(key)
    
    root = os.path.join(root, 'DRIVE') # please change this path to your own请替换为自己的路径
    for name in ["training", "test"]:
        ROOT = os.path.join(root, name)
        register_DRIVE(name, ROOT)

_root = os.getenv("DETECTRON2_DATASETS", "/home/tangwuyang/Dataset/")  # Update this path
register_all_DRIVE(_root)
