from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import os

# 'polyp' is the only category in Kvasir SEG dataset
KVASIR_SEG_CATEGORIES = [{"name": "other", "id": 1, "trainId": 1}, # ,'color': [0, 0, 0],
                         {"name": "polyp", "id": 0, "trainId": 0} # ,'color': [255, 255, 255],
                        ] #

def get_kvasir_seg_meta():
    stuff_ids = [k["id"] for k in KVASIR_SEG_CATEGORIES]
    assert len(stuff_ids) == 2, len(stuff_ids)

    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in KVASIR_SEG_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes
    }
    return ret


def register_kvasir_seg(name, json_file, image_dir, root):
    dataset_name = f"kvasir_seg_sem_seg_{name}"
    image_dir = os.path.join(root, "images")
    gt_dir = os.path.join(root, "masks")
    # 注册数据集
    DatasetCatalog.register(
        dataset_name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="jpg", image_ext="jpg")
    )
    # 设置元数据
    MetadataCatalog.get(dataset_name).set(

        image_root=image_dir,
        sem_seg_root=gt_dir,
        evaluator_type="sem_seg",
        ignore_label=10,
        **get_kvasir_seg_meta(),
    )

def register_all_kvasir_seg(root):
    # 移除已注册的数据集
    # print(DatasetCatalog.list())
    for name in ["train", "val"]:
        key = f"kvasir_seg_sem_seg_{name}"
        if key in  DatasetCatalog.list():
            print(f"Remove registered dataset: {key}")
            DatasetCatalog.remove(key)
    
    root = os.path.join(root, 'kvasir/Kvasir-SEG') # please change this path to your own请替换为自己的路径
    for name, json_file, image_dir in [("train", "train_coco.json", "images"),
                                       ("val", "val_coco.json", "images")]:
        ROOT = os.path.join(root, name)
        json_file = os.path.join(ROOT, json_file)
        image_dir = os.path.join(ROOT, image_dir)
        register_kvasir_seg(name, json_file, image_dir, ROOT)
        # print(DatasetCatalog.get(f"kvasir_seg_sem_seg_train"))

_root = os.getenv("DETECTRON2_DATASETS", "/home/tangwuyang/Dataset/")  # Update this path
register_all_kvasir_seg(_root)
