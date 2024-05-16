import numpy as np
import os
from PIL import Image

def read_and_convert_npz(source_dir, target_dir):
    """
    参数:
    source_dir (str): 包含 .npz 文件的源目录路径。
    target_dir (str): 保存的目标目录路径。
    """
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历源目录中的所有文件
    for file_name in os.listdir(source_dir):
        if file_name.endswith('.npz'):
            file_path = os.path.join(source_dir, file_name)
            with np.load(file_path) as data:
                # 数据包含 'image' 和 'label' 两个键
                image = data['image']
                label = data['label']

                # 保存 image 数据
                image_path = os.path.join(target_dir + "/images/", file_name)
                print(image_path)
                np.savez(image_path, image)

                # 保存 label 数据
                label_path = os.path.join(target_dir + "/labels/", file_name)
                np.savez(label_path, label)


source_directory = '/home/tangwuyang/Dataset/Synapse/train_npz'
target_directory = '/home/tangwuyang/Dataset/Synapse/train'
read_and_convert_npz(source_directory, target_directory)
