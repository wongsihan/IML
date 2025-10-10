# SPINN: Physics-Informed Sparse Neural Network

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)

Official implementation of the paper:

**"Physics-Informed Sparse Neural Network for Permanent Magnet Eddy Current Device Modeling and Analysis"**

*D. Wang, S. Wang, D. Kong, J. Wang, W. Li and M. Pecht*

IEEE Magnetics Letters, vol. 14, pp. 1-5, 2023, Art no. 2100105, doi: 10.1109/LMAG.2023.3288388.

The project has been restructured using a large model. If there are any issues with the code, please contact me.

## Abstract

This paper presents a Physics-Informed Sparse Neural Network (SPINN) method for modeling and analyzing permanent magnet eddy current devices. The method combines the advantages of sparse neural networks with physics-informed constraints to solve electromagnetic field equations efficiently and accurately.

## Installation

```bash
git clone https://github.com/your-username/spinn-official.git
cd spinn-official
pip install -e .
```

## Quick Start

### Official Version (Recommended)
```bash
cd spinn_official
pip install -e .
python examples/cavity_example.py
```

### Original Version
```bash
cd code
python cavity_lubin_slip.py
```

## Usage

```python
from spinn_official import CavityPDE, SPINN2D, Plotter2D, App

# Create and run the application
app = App(CavityPDE, SPINN2D, Plotter2D)
app.run()
```

## Project Structure

- **`spinn_official/`** - Official open source code (recommended)
- **`code/`** - Cleaned core implementation
- **`spinn_paper/`** - Refactored version
- **`spinn_minimal/`** - Minimal version
- **`code_backup/`** - Original backup

## Key Features

- **Physics-Informed**: Incorporates Maxwell equations as constraints
- **Sparse Architecture**: Efficient neural network design with adaptive nodes
- **Multi-Region Support**: Handles complex geometries with multiple materials
- **Automatic Differentiation**: Uses PyTorch's autograd for derivatives
- **Torque Calculation**: Computes electromagnetic torque from field solutions

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

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
