# SPINN: Physics-Informed Sparse Neural Network

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)

This repository contains the official implementation of the paper:

**"Physics-Informed Sparse Neural Network for Permanent Magnet Eddy Current Device Modeling and Analysis"**

*D. Wang, S. Wang, D. Kong, J. Wang, W. Li and M. Pecht*

IEEE Magnetics Letters, vol. 14, pp. 1-5, 2023, Art no. 2100105, doi: 10.1109/LMAG.2023.3288388.

## Abstract

This paper presents a Physics-Informed Sparse Neural Network (SPINN) method for modeling and analyzing permanent magnet eddy current devices. The method combines the advantages of sparse neural networks with physics-informed constraints to solve electromagnetic field equations efficiently and accurately.

## Key Features

- **Physics-Informed**: Incorporates Maxwell equations as constraints
- **Sparse Architecture**: Efficient neural network design with adaptive nodes
- **Multi-Region Support**: Handles complex geometries with multiple materials
- **Automatic Differentiation**: Uses PyTorch's autograd for derivatives
- **Flexible Training**: Supports both Adam and LBFGS optimizers
- **Torque Calculation**: Computes electromagnetic torque from field solutions

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

### Install SPINN

```bash
git clone https://github.com/your-username/spinn-official.git
cd spinn-official
pip install -e .
```

## Quick Start

### Basic Usage

```python
from spinn_official import CavityPDE, SPINN2D, Plotter2D, App

# Create the application
app = App(CavityPDE, SPINN2D, Plotter2D)

# Run with default parameters
app.run()
```

### Run Paper Example

```bash
cd examples
python cavity_example.py
```

This will reproduce the main results from the paper.

### Simple Example

```bash
cd examples
python simple_example.py
```

## Project Structure

```
spinn_official/
├── src/                    # Source code
│   ├── __init__.py
│   ├── core.py            # Core utilities and base classes
│   ├── pde.py             # PDE base classes
│   ├── neural_network.py  # Neural network implementations
│   └── cavity.py          # Main cavity PDE implementation
├── examples/              # Example scripts
│   ├── cavity_example.py  # Main paper example
│   └── simple_example.py  # Simple demonstration
├── tests/                 # Test files
│   └── test_cavity.py     # Unit tests
├── docs/                  # Documentation
├── data/                  # Data files
├── README.md              # This file
├── requirements.txt        # Python dependencies
└── setup.py              # Package setup
```

## Usage

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

### Programmatic Usage

```python
from spinn_official import CavityPDE, SPINN2D, gaussian

# Create PDE
pde = CavityPDE(n_nodes=25, ns=225, nb=25, nbs=25)

# Create neural network
nn = SPINN2D(pde, gaussian, fixed_h=False, use_pu=False)

# Train the network
optimizer = torch.optim.Adam(nn.parameters(), lr=0.001)
for i in range(2500):
    optimizer.zero_grad()
    loss = pde.loss(nn)
    loss.backward()
    optimizer.step()
    
    if i % 500 == 0:
        print(f"Iteration {i}, Loss: {loss.item():.6f}")
```

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

## Testing

Run the test suite:

```bash
python tests/test_cavity.py
```

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China and other funding agencies.
