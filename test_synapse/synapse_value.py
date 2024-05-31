import matplotlib.pyplot as plt
import numpy as np
import os

_root = os.getenv("DETECTRON2_DATASETS", "/home/tangwuyang/Dataset/")  # Update this path
root = os.path.join(_root, "Synapse/train_npz/")

# 读取npz文件
data = np.load(os.path.join(root,"case0030_slice131.npz"))
label = data['label']
image = data['image']
# 显示图像
plt.imshow(label, cmap='gray')
plt.title('Image Visualization')
plt.colorbar()
plt.savefig('./test_synapse/image.png')
