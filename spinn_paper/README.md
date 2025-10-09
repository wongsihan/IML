# SPINN: Physics-Informed Sparse Neural Network

This repository contains the official implementation of the paper:

**"Physics-Informed Sparse Neural Network for Permanent Magnet Eddy Current Device Modeling and Analysis"**

*D. Wang, S. Wang, D. Kong, J. Wang, W. Li and M. Pecht*

IEEE Magnetics Letters, vol. 14, pp. 1-5, 2023, Art no. 2100105, doi: 10.1109/LMAG.2023.3288388.

## Abstract

This paper presents a Physics-Informed Sparse Neural Network (SPINN) method for modeling and analyzing permanent magnet eddy current devices. The method combines the advantages of sparse neural networks with physics-informed constraints to solve electromagnetic field equations efficiently and accurately.

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- SciPy
- Matplotlib

### Install Dependencies

```bash
pip install torch numpy scipy matplotlib
```

## Quick Start

To reproduce the main results from the paper:

```bash
cd examples
python cavity_example.py
```

This will run the cavity PDE problem with the same parameters used in the paper.

## Project Structure

```
spinn_paper/
├── src/                    # Source code
│   ├── __init__.py
│   ├── core.py            # Core utilities and base classes
│   ├── pde.py             # PDE base classes
│   ├── neural_network.py  # Neural network implementations
│   └── cavity_pde.py      # Main cavity PDE implementation
├── examples/              # Example scripts
│   └── cavity_example.py  # Main example
├── data/                  # Data files
├── tests/                 # Test files
└── README.md             # This file
```

## Usage

### Basic Usage

```python
from src import CavityPDE, SPINN2D, Plotter2D, App

# Create the application
app = App(CavityPDE, SPINN2D, Plotter2D)

# Run with default parameters
app.run()
```

### Custom Parameters

```python
# Run with custom parameters
args = [
    '--nodes', '25',           # Number of nodes
    '--samples', '225',        # Number of sample points
    '--n-train', '2500',       # Training iterations
    '--lr', '0.001',           # Learning rate
    '--layers', '3',           # Network layers
    '--neurons', '50',         # Neurons per layer
]

app.run(args)
```

## Key Features

- **Physics-Informed**: Incorporates Maxwell equations as constraints
- **Sparse Architecture**: Efficient neural network design
- **Multi-Region Support**: Handles complex geometries with multiple materials
- **Automatic Differentiation**: Uses PyTorch's autograd for derivatives
- **Flexible Training**: Supports both Adam and LBFGS optimizers

## Physical Model

The implementation solves the electromagnetic field equations in a permanent magnet eddy current device with three regions:

1. **Magnet Region**: Permanent magnet with known magnetic field
2. **Air Gap**: Non-conducting air gap
3. **Copper Region**: Conducting copper with eddy currents

The method uses the magnetic vector potential formulation and solves for the A_dy and A_dz components.

## Results

The method produces:
- Magnetic field distribution
- Eddy current density
- Torque calculation
- Field visualization

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

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
