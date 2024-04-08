from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json, load_sem_seg
import os
import json

# Assuming that 'polyp' is the only category in Kvasir SEG dataset
KVASIR_SEG_CATEGORIES = [{"name": "other", "id": 0, "trainId": 0},
                         {"name": "polyp", "id": 1, "trainId": 1}
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

# def get_kvasir_dicts(root_dir, dataset_name, img_dir, ann_file):
#     dataset_dicts = load_coco_json(ann_file, img_dir, dataset_name)
#     for item in dataset_dicts:
#         # 假设分割标签文件位于'mask'子目录中
#         img_file_name = os.path.basename(item['file_name'])
#         # 根据您的描述，分割文件的文件名和扩展名与图像文件完全相同
#         item['sem_seg_file_name'] = os.path.join(root_dir, 'masks', img_file_name)
#     return dataset_dicts

def get_kvasir_dicts(root_dir, dataset_name, img_dir, ann_file):
    dataset_dicts = load_coco_json(ann_file, img_dir, dataset_name)
    for item in dataset_dicts:
        # 移除annotations字段
        item.pop("annotations", None)
        # 添加分割掩码路径
        img_file_name = os.path.basename(item['file_name'])
        item['sem_seg_file_name'] = os.path.join(root_dir, 'masks', img_file_name)
    return dataset_dicts


def get_kvasir_dicts_custom(root, json_file, image_dir):
    with open(json_file) as f:
        data = json.load(f)

    dataset_dicts = []
    for idx, item in enumerate(data['images']):
        record = {}
        record['file_name'] = os.path.join(image_dir, item['file_name'])
        record['height'] = item['height']
        record['width'] = item['width']
        record['image_id'] = item['id']

        mask_file_name = os.path.join(root, 'masks', item['file_name'])  # 根据您的掩码文件扩展名修改
        record['sem_seg_file_name'] = mask_file_name

        dataset_dicts.append(record)
    return dataset_dicts

def register_kvasir_seg(name, json_file, image_dir, root):
    dataset_name = f"kvasir_seg_sem_seg_{name}"
    # print(get_kvasir_dicts(root, dataset_name, image_dir, json_file))
    # print(get_kvasir_dicts_custom(root, json_file, image_dir))
    image_dir = os.path.join(root, "images")
    gt_dir = os.path.join(root, "masks")
    # 注册数据集
    DatasetCatalog.register(
        dataset_name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="jpg", image_ext="jpg")  # get_kvasir_dicts(root, dataset_name, image_dir, json_file)  #get_kvasir_dicts_custom(root, json_file, image_dir)
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
    
    root = os.path.join(root, 'kvasir/Kvasir-SEG')
    for name, json_file, image_dir in [("train", "train_coco.json", "images"),
                                       ("val", "val_coco.json", "images")]:
        ROOT = os.path.join(root, name)
        json_file = os.path.join(ROOT, json_file)
        image_dir = os.path.join(ROOT, image_dir)
        register_kvasir_seg(name, json_file, image_dir, ROOT)
        # print(DatasetCatalog.get(f"kvasir_seg_sem_seg_train"))

_root = os.getenv("DETECTRON2_DATASETS", "/home/tangwuyang/Dataset/")  # Update this path
register_all_kvasir_seg(_root)
