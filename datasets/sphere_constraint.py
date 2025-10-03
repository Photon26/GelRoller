import numpy as np
import torch
import cv2 as cv
from scipy.optimize import least_squares

def fit_sphere_to_contact(contact_mask, depth_estimation=None):
    """
    从接触区域拟合球体参数
    
    Args:
        contact_mask: 接触区域掩码 (H, W)
        depth_estimation: 可选的深度估计，用于更精确的拟合
    
    Returns:
        sphere_params: (center_x, center_y, center_z, radius)
    """
    # 获取接触区域的像素坐标
    contact_pixels = np.where(contact_mask > 0.5)
    y_coords, x_coords = contact_pixels
    
    if len(x_coords) < 10:  # 需要足够的点进行拟合
        return None
    
    # 如果没有深度信息，假设接触区域在平面上 (z=0)
    if depth_estimation is None:
        z_coords = np.zeros_like(x_coords, dtype=np.float32)
    else:
        z_coords = depth_estimation[contact_pixels]
    
    # 初始估计：假设球心在接触区域中心上方
    center_x_init = np.mean(x_coords)
    center_y_init = np.mean(y_coords)
    center_z_init = np.max(z_coords) + 50  # 假设球心在表面上方50像素
    radius_init = np.sqrt((x_coords - center_x_init)**2 + 
                         (y_coords - center_y_init)**2).max()
    
    initial_params = [center_x_init, center_y_init, center_z_init, radius_init]
    
    def sphere_residual(params, x, y, z):
        cx, cy, cz, r = params
        return (x - cx)**2 + (y - cy)**2 + (z - cz)**2 - r**2
    
    # 使用最小二乘法拟合球体
    result = least_squares(
        sphere_residual, 
        initial_params, 
        args=(x_coords, y_coords, z_coords)
    )
    
    return result.x if result.success else initial_params

def calculate_sphere_normals(sphere_params, contact_mask, image_shape):
    """
    根据球体参数计算接触区域的真实法向量
    
    Args:
        sphere_params: (center_x, center_y, center_z, radius)
        contact_mask: 接触区域掩码
        image_shape: (H, W)
    
    Returns:
        sphere_normals: (H, W, 3) 球体表面法向量
    """
    cx, cy, cz, radius = sphere_params
    H, W = image_shape
    
    # 创建坐标网格
    y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # 初始化法向量数组
    sphere_normals = np.zeros((H, W, 3), dtype=np.float32)
    
    # 只计算接触区域的法向量
    contact_pixels = np.where(contact_mask > 0.5)
    
    for i in range(len(contact_pixels[0])):
        y, x = contact_pixels[0][i], contact_pixels[1][i]
        
        # 计算该点到球心的向量（即法向量方向）
        # 假设接触点在平面上，z坐标为0
        point_to_center = np.array([cx - x, cy - y, cz - 0])
        
        # 归一化得到法向量（指向球心外侧）
        normal = point_to_center / np.linalg.norm(point_to_center)
        
        # 确保法向量指向相机方向（z分量为负）
        if normal[2] > 0:
            normal = -normal
            
        sphere_normals[y, x] = normal
    
    return sphere_normals

def create_sphere_mask(sphere_params, image_shape, contact_center, contact_radius):
    """
    根据球体参数和接触信息创建球体区域掩码
    
    Args:
        sphere_params: (center_x, center_y, center_z, radius)
        image_shape: (H, W)
        contact_center: (x, y) 接触区域中心
        contact_radius: 接触区域半径（像素）
    
    Returns:
        sphere_mask: (H, W) 球体接触区域掩码
    """
    H, W = image_shape
    cx_contact, cy_contact = contact_center
    
    # 创建圆形掩码
    y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    distance_from_center = np.sqrt((x_grid - cx_contact)**2 + (y_grid - cy_contact)**2)
    
    sphere_mask = (distance_from_center <= contact_radius).astype(np.float32)
    
    return sphere_mask

def get_sphere_constraint_data(image_path, sphere_center, sphere_radius, contact_center, contact_radius):
    """
    为球体约束准备数据
    
    Args:
        image_path: 图像路径
        sphere_center: (x, y, z) 球心坐标
        sphere_radius: 球体半径
        contact_center: (x, y) 接触区域中心
        contact_radius: 接触区域半径
    
    Returns:
        constraint_data: 包含球体约束信息的字典
    """
    # 读取图像
    image = cv.imread(image_path)[:, :, ::-1].astype(np.float32) / 255.
    H, W = image.shape[:2]
    
    # 创建球体接触区域掩码
    sphere_mask = create_sphere_mask(
        (sphere_center[0], sphere_center[1], sphere_center[2], sphere_radius),
        (H, W), contact_center, contact_radius
    )
    
    # 计算球体表面法向量
    sphere_normals = calculate_sphere_normals(
        (sphere_center[0], sphere_center[1], sphere_center[2], sphere_radius),
        sphere_mask, (H, W)
    )
    
    # 创建有效区域掩码（整个图像都有效）
    valid_mask = np.ones((H, W), dtype=np.float32)
    
    # 创建背景掩码（平面区域，法向量为[0,0,1]）
    background_mask = 1.0 - sphere_mask
    background_normals = np.zeros((H, W, 3), dtype=np.float32)
    background_normals[:, :, 2] = 1.0  # 平面法向量指向z轴正方向
    
    return {
        'image': image,
        'sphere_mask': sphere_mask,           # 球体接触区域
        'sphere_normals': sphere_normals,     # 球体区域的真实法向量
        'background_mask': background_mask,   # 背景平面区域
        'background_normals': background_normals,  # 背景区域的真实法向量
        'valid_mask': valid_mask,            # 有效重建区域
        'sphere_params': (sphere_center[0], sphere_center[1], sphere_center[2], sphere_radius)
    }
