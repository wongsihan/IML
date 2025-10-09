# SPINN: Physics-Informed Sparse Neural Network

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)

Official implementation of the paper:

**"Physics-Informed Sparse Neural Network for Permanent Magnet Eddy Current Device Modeling and Analysis"**

*D. Wang, S. Wang, D. Kong, J. Wang, W. Li and M. Pecht*

IEEE Magnetics Letters, vol. 14, pp. 1-5, 2023, Art no. 2100105, doi: 10.1109/LMAG.2023.3288388.

## Project Structure

This repository contains multiple versions of the SPINN implementation:

### 📁 **spinn_official/** - Official Open Source Code (Recommended)
The clean, modular, and well-documented official implementation:
```
spinn_official/
├── src/                    # Core source code
├── examples/              # Example scripts
├── tests/                 # Test files
├── README.md              # Complete documentation
├── requirements.txt       # Dependencies
└── setup.py              # Package installation
```

**Quick Start:**
```bash
cd spinn_official
pip install -e .
python examples/cavity_example.py
```

### 📁 **code/** - Cleaned Core Implementation
The original implementation after cleanup (6 core files only):
```
code/
├── cavity_lubin_slip.py      # Main paper implementation
├── common_nomayavi.py        # Utility functions
├── spinn2d_nomayavi.py       # Neural network
├── pde2d_base_nomayavi.py    # PDE base class
└── data/                     # Data files
```

**Quick Start:**
```bash
cd code
python cavity_lubin_slip.py
```

### 📁 **spinn_paper/** - Refactored Version
A refactored version with modular structure:
```
spinn_paper/
├── src/                    # Modular source code
├── examples/              # Example scripts
└── tests/                 # Test files
```

### 📁 **spinn_minimal/** - Minimal Version
A minimal version with only essential files:
```
spinn_minimal/
├── cavity_lubin_slip.py      # Main implementation
├── common_nomayavi.py        # Utilities
├── spinn2d_nomayavi.py       # Neural network
└── pde2d_base_nomayavi.py    # PDE base
```

### 📁 **code_backup/** - Original Backup
Complete backup of the original code (94 files) before cleanup.

## Key Features

- **Physics-Informed**: Incorporates Maxwell equations as constraints
- **Sparse Architecture**: Efficient neural network design with adaptive nodes
- **Multi-Region Support**: Handles complex geometries with multiple materials
- **Automatic Differentiation**: Uses PyTorch's autograd for derivatives
- **Torque Calculation**: Computes electromagnetic torque from field solutions

## Installation

### For Official Version (Recommended)
```bash
cd spinn_official
pip install -e .
```

### For Original Version
```bash
cd code
pip install -r ../requirements.txt
```

## Usage

### Official Version (Recommended)
```python
from spinn_official import CavityPDE, SPINN2D, Plotter2D, App

# Create and run the application
app = App(CavityPDE, SPINN2D, Plotter2D)
app.run()
```

### Cleaned Core Version
```bash
cd code
python cavity_lubin_slip.py
```

### Refactored Version
```bash
cd spinn_paper
python examples/cavity_example.py
```

### Minimal Version
```bash
cd spinn_minimal
python cavity_lubin_slip.py
```

## Reproducing Results

To reproduce the main results from the paper:

1. **Official Version (Recommended):**
   ```bash
   cd spinn_official
   pip install -e .
   python examples/cavity_example.py
   ```

2. **Cleaned Core Version:**
   ```bash
   cd code
   python cavity_lubin_slip.py
   ```

3. **Refactored Version:**
   ```bash
   cd spinn_paper
   python examples/cavity_example.py
   ```

4. **Minimal Version:**
   ```bash
   cd spinn_minimal
   python cavity_lubin_slip.py
   ```

## Physical Model

The implementation solves electromagnetic field equations in a permanent magnet eddy current device with three regions:

1. **Magnet Region**: Permanent magnet with known magnetic field
2. **Air Gap**: Non-conducting air gap  
3. **Copper Region**: Conducting copper with eddy currents

## Citation

If you use this code in your research, please cite:

```bibtex
@article{wang2023physics,
  title={Physics-Informed Sparse Neural Network for Permanent Magnet Eddy Current Device Modeling and Analysis},
  author={Wang, D. and Wang, S. and Kong, D. and Wang, J. and Li, W. and Pecht, M.},
  journal={IEEE Magnetics Letters},
  volume={14},
  pages={1--5},
  year={2023},
  publisher={IEEE},
  doi={10.1109/LMAG.2023.3288388}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](spinn_official/LICENSE) file for details.

## Project History

- **Original Code** (`code_backup/`): 94 files with duplicates and unused code
- **Cleaned Code** (`code/`): 6 core files after cleanup
- **Minimal Code** (`spinn_minimal/`): Essential files only
- **Refactored Code** (`spinn_paper/`): Modular structure with documentation
- **Official Code** (`spinn_official/`): Production-ready open source implementation

## Version Comparison

| Version | Files | Structure | Documentation | Recommended |
|---------|-------|-----------|---------------|-------------|
| Original | 94 | Messy | None | ❌ |
| Cleaned | 6 | Simple | Basic | ✅ |
| Minimal | 4 | Simple | Basic | ✅ |
| Refactored | 9 | Modular | Good | ✅ |
| Official | 9 | Modular | Complete | ✅ |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.