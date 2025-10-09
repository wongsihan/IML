# SPINN - 清理后的核心实现

这是SPINN论文的核心实现，已经删除了所有重复和无用的文件。

## 核心文件

- `cavity_lubin_slip.py` - **主要论文实现**，包含三区域电磁场建模
- `common_nomayavi.py` - 通用工具函数和设备管理
- `spinn2d_nomayavi.py` - 2D稀疏神经网络实现
- `pde2d_base_nomayavi.py` - 2D PDE基类
- `data/` - 数据文件目录

## 使用方法

运行主要论文结果：

```bash
python cavity_lubin_slip.py
```

## 论文引用

D. Wang, S. Wang, D. Kong, J. Wang, W. Li and M. Pecht, "Physics-Informed Sparse Neural Network for Permanent Magnet Eddy Current Device Modeling and Analysis," in IEEE Magnetics Letters, vol. 14, pp. 1-5, 2023, Art no. 2100105, doi: 10.1109/LMAG.2023.3288388.

## 清理说明

已删除的文件：
- 重复的common.py, cavity.py等实现
- 与论文无关的1D实现（ode1.py, ode2.py等）
- 重复的PDE实现
- 旧的保存目录和临时文件
- 与主要论文无关的示例

现在项目结构简洁，只包含论文核心实现。
