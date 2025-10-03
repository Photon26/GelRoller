import cv2 as cv
import os
import numpy as np
import torch
import glob
import scipy.io as sio
from .sphere_constraint import get_sphere_constraint_data

def load_planar_sensor(path, cfg=None):
    """
    加载平面传感器数据，支持球体约束
    
    Args:
        path: 数据路径
        cfg: 配置文件，包含球体参数
    
    Returns:
        out_dict: 包含图像和约束信息的字典
    """
    images = []
    for img_file in sorted(glob.glob(os.path.join(path, "[0-9]*.png"))):
        img = cv.imread(img_file)[:, :, ::-1].astype(np.float32) / 255.
        img = cv.GaussianBlur(img, (5, 5), 0)
        images.append(img)
    images = np.stack(images, axis=0)
    
    # 检查是否有传统的valid和mask文件
    valid_files = os.path.join(path, "valid.png")
    mask_files = os.path.join(path, "mask.png")
    
    if os.path.exists(valid_files) and os.path.exists(mask_files):
        # 使用传统方式加载
        valid_region = cv.imread(valid_files, 0).astype(np.float32) / 255.
        mask = cv.imread(mask_files, 0).astype(np.float32) / 255.
        mask = cv.GaussianBlur(mask, (5, 5), 0)
        
        # 平面传感器的背景法向量都是[0,0,1]
        H, W = images.shape[1:3]
        gt_normal = np.zeros((H, W, 3), dtype=np.float32)
        gt_normal[:, :, 2] = 1.0  # z方向法向量
        
        # 平面传感器的深度都是0
        gt_z = np.zeros((H, W), dtype=np.float32)
        
    elif hasattr(cfg, 'sphere_constraint') and cfg.sphere_constraint.enabled:
        # 使用球体约束方式
        sphere_cfg = cfg.sphere_constraint
        
        # 获取球体约束数据
        constraint_data = get_sphere_constraint_data(
            image_path=sorted(glob.glob(os.path.join(path, "[0-9]*.png")))[0],
            sphere_center=sphere_cfg.sphere_center,
            sphere_radius=sphere_cfg.sphere_radius,
            contact_center=sphere_cfg.contact_center,
            contact_radius=sphere_cfg.contact_radius
        )
        
        valid_region = constraint_data['valid_mask']
        
        # 创建复合掩码：球体区域=1，背景区域=0，其他接触区域=2
        mask = constraint_data['sphere_mask'].copy()
        
        # 如果有其他接触区域的掩码文件，也加载进来
        other_contact_file = os.path.join(path, "other_contact.png")
        if os.path.exists(other_contact_file):
            other_contact = cv.imread(other_contact_file, 0).astype(np.float32) / 255.
            mask[other_contact > 0.5] = 2.0  # 标记为其他接触区域
        
        # 创建复合法向量：球体区域使用计算的法向量，背景使用[0,0,1]
        gt_normal = constraint_data['background_normals'].copy()
        sphere_pixels = constraint_data['sphere_mask'] > 0.5
        gt_normal[sphere_pixels] = constraint_data['sphere_normals'][sphere_pixels]
        
        # 平面传感器的深度信息
        gt_z = np.zeros_like(constraint_data['sphere_mask'])
        
    else:
        raise ValueError("需要提供valid.png和mask.png文件，或者在配置中启用sphere_constraint")
    
    # 处理稀疏输入（如果配置了）
    if hasattr(cfg, 'dataset') and hasattr(cfg.dataset, 'sparse_input_random_seed') and hasattr(cfg.dataset, 'sparse_input'):
        if cfg.dataset.sparse_input_random_seed is not None and cfg.dataset.sparse_input is not None:
            np.random.seed(cfg.dataset.sparse_input_random_seed)
            select_idx = np.random.permutation(len(images))[:cfg.dataset.sparse_input]
            print('Random seed: %d .   Selected random index: ' % cfg.dataset.sparse_input_random_seed, select_idx)
            images = images[select_idx]
    
    out_dict = {
        'images': images, 
        'valid_region': valid_region, 
        'mask': mask, 
        'gt_normal': gt_normal, 
        'gt_z': gt_z
    }
    
    # 如果使用球体约束，添加额外信息
    if hasattr(cfg, 'sphere_constraint') and cfg.sphere_constraint.enabled:
        out_dict.update({
            'sphere_mask': constraint_data['sphere_mask'],
            'sphere_normals': constraint_data['sphere_normals'],
            'background_mask': constraint_data['background_mask'],
            'sphere_params': constraint_data['sphere_params']
        })
    
    return out_dict
