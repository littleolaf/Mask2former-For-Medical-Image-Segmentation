import h5py
import numpy as np
import os

_root = os.getenv("DETECTRON2_DATASETS", "/home/tangwuyang/Dataset/")  # Update this path
root = os.path.join(_root, "Synapse/test_vol_h5/")
# 设定输入文件夹和输出文件夹
input_dir = root
output_dir_images = os.path.join(_root, "Synapse/val/images")
output_dir_labels = os.path.join(_root, "Synapse/val/labels")
slice_images = os.path.join(_root, "Synapse/test/images")
slice_labels = os.path.join(_root, "Synapse/test/labels")

# 创建输出文件夹
os.makedirs(output_dir_images, exist_ok=True)
os.makedirs(output_dir_labels, exist_ok=True)

def h5_to_npz(input_dir, output_dir_images, output_dir_labels):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.npy.h5'):
            # 构造完整的文件路径
            file_path = os.path.join(input_dir, filename)
            
            # 读取HDF5文件
            with h5py.File(file_path, 'r') as file:
                # 检查文件中是否存在'image'和'label'数据集
                if 'image' in file and 'label' in file:
                    # 读取数据
                    image_data = file['image'][...]
                    label_data = file['label'][...]

                    # 构造输出文件名
                    output_filename_image = filename.replace('.npy.h5', '.npz')
                    output_filename_label = filename.replace('.npy.h5', '.npz')

                    # 保存数据到不同的路径
                    np.savez(os.path.join(output_dir_images, output_filename_image), data=image_data)
                    np.savez(os.path.join(output_dir_labels, output_filename_label), data=label_data)


def process_and_save_slices(data, base_filename, output_dir):
    # 遍历三维数组的第一个维度，即不同的切片
    for i, slice in enumerate(data):
        # 生成每个切片的文件名，格式为"case00*_slice00*.npz"
        if slice.ndim == 2: # 判断仅保存二维数组
            slice_filename = f'{base_filename}_slice{i:03}.npz'
            # 保存切片
            np.savez(os.path.join(output_dir, slice_filename), slice)

def split_and_save(input_dir, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历input_dir中的所有npz文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.npz'):
            # 构建完整的文件路径
            file_path = os.path.join(input_dir, filename)
            # 加载npz文件
            with np.load(file_path) as data:
                # 提取数组
                array = data['data']
                # 获取不带扩展名的文件名
                base_filename = os.path.splitext(filename)[0]
                # 处理并保存每个切片
                process_and_save_slices(array, base_filename, output_dir)

# 处理图片和标签
h5_to_npz(input_dir, output_dir_images, output_dir_labels)
# 切片
split_and_save(output_dir_labels, slice_labels)
split_and_save(output_dir_images, slice_images)
