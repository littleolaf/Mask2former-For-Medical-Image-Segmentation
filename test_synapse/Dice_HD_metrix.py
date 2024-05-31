from medpy.metric import binary
import numpy as np

def calculate_metrics(predict, ground_truth):
    """
    计算多分类任务的 Dice 相似系数和 Hausdorff 距离。
    
    参数:
    predict (numpy.ndarray): 预测矩阵，形状为 [512, 512]，值为预测的类别标签。
    ground_truth (numpy.ndarray): 真实标签矩阵，形状为 [512, 512]，值为真实的类别标签。

    返回:
    dict: 包含每个类别的 Dice 相似系数和 Hausdorff 距离。
    """
    # 获取 predict 和 ground_truth 中的标签并集，去掉标签0
    unique_labels = np.setdiff1d(np.unique(np.concatenate((predict, ground_truth))), [0])
    
    dice_scores = {}
    hausdorff_distances = {}
    
    for class_id in unique_labels:
        # 创建当前类别的二值化预测和真实标签
        predict_binary = (predict == class_id).astype(np.uint8)
        ground_truth_binary = (ground_truth == class_id).astype(np.uint8)
        
        # 判断该类别是否在两者中都存在
        if np.any(predict_binary) and np.any(ground_truth_binary):
            # 计算 Dice 相似系数
            dice = binary.dc(ground_truth_binary, predict_binary)
            
            # 计算 Hausdorff 距离
            try:
                hd = binary.hd(ground_truth_binary, predict_binary)
            except RuntimeError:
                hd = float('inf')
        else:
            dice = 0.0
            hd = 0.0
        
        # 存储结果
        dice_scores[class_id] = dice
        hausdorff_distances[class_id] = hd
    
    return {
        "dice_scores": dice_scores,
        "hausdorff_distances": hausdorff_distances
    }

# 示例用法
predict = np.random.randint(0, 5, (512, 512))  # 假设有5个类别
ground_truth = np.random.randint(0, 5, (512, 512))

# metrics = calculate_metrics(predict, ground_truth)
# print("Dice scores:", metrics["dice_scores"])
# print("Hausdorff distances:", metrics["hausdorff_distances"])

from medpy.metric import binary
import numpy as np

def calculate_metrics_from_confusion_matrix(conf_matrix):
    """
    计算多分类任务的 Dice 相似系数和 Hausdorff 距离。

    参数:
    conf_matrix (numpy.ndarray): 混淆矩阵，形状为 [num_classes, num_classes]。

    返回:
    dict: 包含每个类别的 Dice 相似系数和 Hausdorff 距离。
    """
    num_classes = conf_matrix.shape[0]
    
    dice_scores = {}
    hausdorff_distances = {}
    
    for class_id in range(num_classes):
        # 提取当前类别的二值化预测和真实标签
        tp = conf_matrix[class_id, class_id]  # True Positives
        fn = np.sum(conf_matrix[class_id, :]) - tp  # False Negatives
        fp = np.sum(conf_matrix[:, class_id]) - tp  # False Positives
        
        predict_binary = np.zeros_like(conf_matrix)
        ground_truth_binary = np.zeros_like(conf_matrix)
        
        predict_binary[class_id, :] = conf_matrix[class_id, :]
        ground_truth_binary[:, class_id] = conf_matrix[:, class_id]
        
        # 计算 Dice 相似系数
        if tp + fn + fp > 0:
            dice = 2 * tp / (2 * tp + fn + fp)
        else:
            dice = 0.0
        
        # 计算 Hausdorff 距离
        if tp > 0:
            predict_binary = (predict_binary.flatten() > 0).astype(np.uint8)
            ground_truth_binary = (ground_truth_binary.flatten() > 0).astype(np.uint8)
            try:
                hd = binary.hd(ground_truth_binary, predict_binary)
            except RuntimeError:
                hd = float('inf')
        else:
            hd = 0.0
        
        # 存储结果
        dice_scores[class_id] = dice
        hausdorff_distances[class_id] = hd
    
    return {
        "dice_scores": dice_scores,
        "hausdorff_distances": hausdorff_distances
    }

# 示例用法
conf_matrix = np.array([
    [50,  2,  1],
    [ 5, 45,  0],
    [ 2,  3, 40]
])

# metrics = calculate_metrics_from_confusion_matrix(conf_matrix)
# print("Dice scores:", metrics["dice_scores"])
# print("Hausdorff distances:", metrics["hausdorff_distances"])
