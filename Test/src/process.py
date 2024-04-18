import cv2
import numpy as np


def remove_small_objects_per_class(mask, min_size=100):
    """
    对每个类别独立移除小对象。

    参数:
    - mask: 分割结果的掩码，其中0为背景，1~11为不同的前景类别。
    - min_size: 被认为是小对象的像素数量阈值。

    返回:
    - mask: 经过处理后的掩码。
    """
    output_mask = np.zeros_like(mask)

    # 处理每个前景类别
    for class_id in range(1, 12):  # 遍历1到11的类别ID
        # 获取当前类别的掩码
        class_mask = np.where(mask == class_id, 1, 0).astype(np.uint8)

        # 找到所有连通区域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)

        # 移除小对象
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                output_mask[labels == i] = class_id

    return output_mask


def open_close_per_class(mask, kernel_size=3, operation='open'):
    """
    对多分类分割结果的每个类别执行开运算或闭运算。

    参数:
    - mask: 分割掩码，值为类别索引。
    - kernel_size: 核大小。
    - operation: 'open' 或 'close'。

    返回:
    - 处理后的分割掩码。
    """
    output_mask = np.zeros_like(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    unique_classes = np.unique(mask)
    for cls in unique_classes:
        if cls == 0:  # 跳过背景
            continue
        class_mask = (mask == cls).astype(np.uint8)

        if operation == 'open':
            processed_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)
        elif operation == 'close':
            processed_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)

        # 将处理后的类别掩码回填到输出掩码中
        output_mask[processed_mask == 1] = cls

    return output_mask
