#!/usr/bin/env python3
"""
创建球体约束数据的辅助脚本
用于从平面传感器图像中提取球体接触区域并生成约束数据
"""

import cv2 as cv
import numpy as np
import os
import argparse
from datasets.sphere_constraint import get_sphere_constraint_data
import matplotlib.pyplot as plt

def interactive_sphere_selection(image_path):
    """
    交互式选择球体接触区域
    
    Args:
        image_path: 输入图像路径
    
    Returns:
        sphere_params: (center_x, center_y, center_z, radius)
        contact_params: (center_x, center_y, radius)
    """
    # 读取图像
    image = cv.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    print("请在图像中点击球体接触区域的中心...")
    
    # 全局变量存储点击位置
    click_points = []
    
    def on_click(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            click_points.append((x, y))
            # 在图像上标记点击位置
            cv.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv.imshow('Select Sphere Center', image)
            print(f"选择的中心点: ({x}, {y})")
    
    # 显示图像并等待点击
    cv.imshow('Select Sphere Center', image)
    cv.setMouseCallback('Select Sphere Center', on_click)
    
    print("点击球体接触区域的中心，然后按任意键继续...")
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    if not click_points:
        raise ValueError("未选择中心点")
    
    center_x, center_y = click_points[-1]  # 使用最后一个点击位置
    
    # 输入球体参数
    print(f"选择的接触中心: ({center_x}, {center_y})")
    
    try:
        contact_radius = float(input("请输入接触区域半径（像素）[默认30]: ") or "30")
        sphere_radius = float(input("请输入球体半径（像素）[默认50]: ") or "50")
        sphere_z = float(input("请输入球心z坐标（像素，相对于平面）[默认100]: ") or "100")
    except ValueError:
        print("使用默认参数")
        contact_radius = 30
        sphere_radius = 50
        sphere_z = 100
    
    sphere_params = (center_x, center_y, sphere_z, sphere_radius)
    contact_params = (center_x, center_y, contact_radius)
    
    return sphere_params, contact_params

def visualize_sphere_constraint(image_path, sphere_params, contact_params, output_dir):
    """
    可视化球体约束结果
    """
    # 获取约束数据
    constraint_data = get_sphere_constraint_data(
        image_path, 
        sphere_params[:3], 
        sphere_params[3],
        contact_params[:2], 
        contact_params[2]
    )
    
    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原始图像
    axes[0, 0].imshow(constraint_data['image'])
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 球体掩码
    axes[0, 1].imshow(constraint_data['sphere_mask'], cmap='gray')
    axes[0, 1].set_title('Sphere Mask')
    axes[0, 1].axis('off')
    
    # 背景掩码
    axes[0, 2].imshow(constraint_data['background_mask'], cmap='gray')
    axes[0, 2].set_title('Background Mask')
    axes[0, 2].axis('off')
    
    # 球体法向量（可视化）
    sphere_normal_vis = (constraint_data['sphere_normals'] + 1) / 2
    axes[1, 0].imshow(sphere_normal_vis)
    axes[1, 0].set_title('Sphere Normals')
    axes[1, 0].axis('off')
    
    # 背景法向量（可视化）
    bg_normal_vis = (constraint_data['background_normals'] + 1) / 2
    axes[1, 1].imshow(bg_normal_vis)
    axes[1, 1].set_title('Background Normals')
    axes[1, 1].axis('off')
    
    # 合成掩码
    composite_mask = np.zeros_like(constraint_data['sphere_mask'])
    composite_mask[constraint_data['sphere_mask'] > 0.5] = 1.0  # 球体区域
    composite_mask[constraint_data['background_mask'] > 0.5] = 0.5  # 背景区域
    axes[1, 2].imshow(composite_mask, cmap='viridis')
    axes[1, 2].set_title('Composite Mask\n(Yellow=Sphere, Dark=Background)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # 保存可视化结果
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'sphere_constraint_visualization.png'), dpi=150, bbox_inches='tight')
    print(f"可视化结果保存到: {os.path.join(output_dir, 'sphere_constraint_visualization.png')}")
    
    # 保存数据文件
    np.save(os.path.join(output_dir, 'sphere_mask.npy'), constraint_data['sphere_mask'])
    np.save(os.path.join(output_dir, 'sphere_normals.npy'), constraint_data['sphere_normals'])
    np.save(os.path.join(output_dir, 'background_mask.npy'), constraint_data['background_mask'])
    np.save(os.path.join(output_dir, 'background_normals.npy'), constraint_data['background_normals'])
    
    # 保存掩码图像（用于传统加载方式）
    cv.imwrite(os.path.join(output_dir, 'sphere_mask.png'), 
               (constraint_data['sphere_mask'] * 255).astype(np.uint8))
    cv.imwrite(os.path.join(output_dir, 'valid.png'), 
               (constraint_data['valid_mask'] * 255).astype(np.uint8))
    
    # 创建复合掩码文件
    composite_mask_file = np.zeros_like(constraint_data['sphere_mask'])
    composite_mask_file[constraint_data['sphere_mask'] > 0.5] = 255  # 球体区域标记为255
    cv.imwrite(os.path.join(output_dir, 'mask.png'), composite_mask_file.astype(np.uint8))
    
    print(f"数据文件保存到: {output_dir}")
    
    return constraint_data

def main():
    parser = argparse.ArgumentParser(description='创建球体约束数据')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--interactive', action='store_true', help='交互式选择球体区域')
    parser.add_argument('--sphere_center', type=float, nargs=3, help='球心坐标 (x, y, z)')
    parser.add_argument('--sphere_radius', type=float, help='球体半径')
    parser.add_argument('--contact_center', type=float, nargs=2, help='接触中心 (x, y)')
    parser.add_argument('--contact_radius', type=float, help='接触半径')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        raise ValueError(f"图像文件不存在: {args.image}")
    
    if args.interactive:
        # 交互式模式
        sphere_params, contact_params = interactive_sphere_selection(args.image)
    else:
        # 命令行参数模式
        if not all([args.sphere_center, args.sphere_radius, args.contact_center, args.contact_radius]):
            raise ValueError("非交互模式需要提供所有球体参数")
        
        sphere_params = (*args.sphere_center, args.sphere_radius)
        contact_params = (*args.contact_center, args.contact_radius)
    
    print(f"球体参数: {sphere_params}")
    print(f"接触参数: {contact_params}")
    
    # 生成约束数据并可视化
    constraint_data = visualize_sphere_constraint(args.image, sphere_params, contact_params, args.output)
    
    # 生成配置文件模板
    config_template = f"""# 球体约束配置（添加到你的配置文件中）
sphere_constraint:
  enabled: true
  sphere_center: [{sphere_params[0]}, {sphere_params[1]}, {sphere_params[2]}]
  sphere_radius: {sphere_params[3]}
  contact_center: [{contact_params[0]}, {contact_params[1]}]
  contact_radius: {contact_params[2]}
"""
    
    with open(os.path.join(args.output, 'sphere_config.yml'), 'w') as f:
        f.write(config_template)
    
    print(f"配置模板保存到: {os.path.join(args.output, 'sphere_config.yml')}")
    print("完成！你可以将配置模板中的内容添加到你的训练配置文件中。")

if __name__ == '__main__':
    main()
