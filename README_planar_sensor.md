# 平面传感器球体约束方法

本文档说明如何将GelRoller的方法适配到平面触觉传感器，使用球体接触约束进行3D重建。

## 核心思想

### 原始GelRoller方法
- **背景区域**：已知曲面几何 → 已知法向量 → 约束学习
- **接触区域**：未知形状 → 待重建

### 平面传感器球体约束方法  
- **背景区域**：平面几何 → 法向量恒定`[0,0,1]` → 约束信息有限
- **球体接触区域**：已知球面几何 → 可计算法向量 → 强约束学习
- **其他接触区域**：未知形状 → 待重建

## 使用步骤

### 1. 准备数据

#### 方法一：交互式创建球体约束数据
```bash
python create_sphere_data.py --image your_image.png --output data/planar_ball --interactive
```

#### 方法二：命令行指定参数
```bash
python create_sphere_data.py \
    --image your_image.png \
    --output data/planar_ball \
    --sphere_center 320 240 100 \
    --sphere_radius 50 \
    --contact_center 320 240 \
    --contact_radius 30
```

### 2. 配置训练参数

编辑 `configs/planar_sensor.yml`：

```yaml
dataset:
  data_path: data/planar_ball

sphere_constraint:
  enabled: true
  sphere_center: [320, 240, 100]  # 根据实际情况调整
  sphere_radius: 50
  contact_center: [320, 240]
  contact_radius: 30

loss:
  sphere_normal_factor: 1.0      # 球体法向量约束权重
  sphere_ph_factor: 2.0          # 球体光度约束权重  
  background_normal_factor: 0.2  # 背景约束权重（较小）
  other_contact_factor: 3.0      # 其他接触区域权重
```

### 3. 开始训练

```bash
python train_planar.py --config configs/planar_sensor.yml
```

## 新的损失函数设计

### 球体约束模式（推荐）

1. **球体区域法向量约束**
   ```python
   sphere_normal_loss = |est_normal - gt_sphere_normal| * sphere_mask
   ```

2. **球体区域光度约束**
   ```python
   sphere_ph_loss = |render_rgb - gt_rgb| * sphere_mask
   ```

3. **背景区域约束**（可选，权重较小）
   ```python
   bg_normal_loss = |est_normal - [0,0,1]| * background_mask
   ```

4. **其他接触区域视觉一致性约束**
   ```python
   other_contact_loss = L1_loss + SSIM_loss
   ```

### 传统模式（兼容性）

当 `sphere_constraint.enabled = false` 时，使用原始的背景+接触约束。

## 文件结构

```
gelroller/
├── datasets/
│   ├── sphere_constraint.py      # 球体约束计算
│   ├── load_planar_sensor.py     # 平面传感器数据加载
│   └── planar_dataloader.py      # 平面传感器数据加载器
├── configs/
│   └── planar_sensor.yml         # 平面传感器配置
├── train_planar.py               # 平面传感器训练脚本
├── create_sphere_data.py         # 球体数据创建工具
└── README_planar_sensor.md       # 本文档
```

## 关键参数说明

### 球体参数
- `sphere_center`: 球心3D坐标，z坐标表示球心相对于平面的高度
- `sphere_radius`: 球体半径（像素单位）
- `contact_center`: 接触区域在图像中的中心坐标
- `contact_radius`: 接触区域半径（像素单位）

### 损失权重
- `sphere_normal_factor`: 球体法向量约束权重，建议1.0
- `sphere_ph_factor`: 球体光度约束权重，建议2.0
- `background_normal_factor`: 背景约束权重，建议0.2（因为平面背景信息有限）
- `other_contact_factor`: 其他接触区域权重，建议3.0

## 优势与适用场景

### 优势
1. **强几何约束**：球体几何已知，提供可靠的法向量约束
2. **自监督学习**：不需要接触区域的真实标注
3. **灵活扩展**：可以处理球体+其他未知物体的复合接触
4. **物理合理**：基于真实的球体几何和光照物理模型

### 适用场景
1. 平面触觉传感器的3D重建
2. 已知几何物体（球、圆柱等）的接触重建
3. 需要高精度法向量估计的应用
4. 自监督触觉学习研究

## 注意事项

1. **球体参数标定**：需要准确测量或标定球体的实际参数
2. **接触区域检测**：确保球体接触区域的掩码准确
3. **光照条件**：保证接触区域有足够的光照变化
4. **传感器标定**：需要知道像素与物理尺寸的对应关系

## 扩展方向

1. **多球体约束**：使用多个不同大小的球体提供更多约束
2. **其他几何体**：扩展到圆柱、立方体等已知几何体
3. **动态约束**：结合时序信息进行动态约束
4. **混合约束**：结合深度传感器等其他模态信息

## 故障排除

### 常见问题

1. **球体参数不准确**
   - 使用交互式工具重新标定
   - 检查球体半径和中心位置

2. **约束权重不平衡**
   - 调整损失函数权重
   - 监控各项损失的数值范围

3. **接触区域检测错误**
   - 检查掩码文件的准确性
   - 调整接触半径参数

4. **收敛困难**
   - 降低学习率
   - 增加球体约束权重
   - 检查数据质量

如有问题，请参考原始GelRoller论文或联系开发者。
