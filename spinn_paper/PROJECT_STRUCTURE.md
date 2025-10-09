# SPINN Project Structure

This document describes the refactored project structure for the SPINN paper implementation.

## New Structure

```
spinn_paper/
├── src/                          # Core source code
│   ├── __init__.py              # Package initialization
│   ├── core.py                  # Core utilities and base classes
│   ├── pde.py                   # PDE base classes
│   ├── neural_network.py        # Neural network implementations
│   └── cavity_pde.py            # Main cavity PDE implementation
├── examples/                     # Example scripts
│   ├── cavity_example.py        # Main paper example
│   └── simple_example.py        # Simple demonstration
├── tests/                        # Test files
│   └── test_cavity.py           # Unit tests
├── data/                         # Data files (if needed)
├── README.md                     # Main documentation
├── requirements.txt              # Python dependencies
├── setup.py                     # Package setup
├── cleanup_duplicates.py        # Cleanup script
└── PROJECT_STRUCTURE.md         # This file
```

## Key Improvements

### 1. **Eliminated Duplicates**
- Removed duplicate files:
  - `common.py` vs `common_nomayavi.py` → kept `common_nomayavi.py`
  - Multiple cavity implementations → kept only the main one
  - Multiple spinn implementations → kept the nomayavi version

### 2. **Clean Architecture**
- **Core module** (`src/core.py`): Base classes for PDE, Plotter, Optimizer, App
- **PDE module** (`src/pde.py`): PDE base classes and utilities
- **Neural Network module** (`src/neural_network.py`): SPINN implementation
- **Cavity PDE** (`src/cavity_pde.py`): Main paper implementation

### 3. **Removed Unnecessary Files**
- Old 1D implementations not relevant to the paper
- Duplicate cavity implementations
- Test files not related to the main paper
- Old save directories and temporary files

### 4. **Maintained Paper Fidelity**
- Kept the main implementation from `cavity_lubin_slip.py`
- Preserved all physics equations and parameters
- Maintained the three-region model (magnet, air gap, copper)

## Original vs Refactored

### Original Structure (Messy)
```
code/
├── common.py                    # Duplicate
├── common_nomayavi.py          # Keep
├── cavity.py                   # Duplicate
├── cavity_lubin_slip.py        # Main implementation
├── cavity_pmd_lubin.py         # Duplicate
├── cavity_pmd_lubin_nomayavi.py # Duplicate
├── ... (many more duplicates)
├── spinn2d.py                  # Duplicate
├── spinn2d_nomayavi.py         # Keep
├── pde2d_base.py              # Duplicate
├── pde2d_base_nomayavi.py     # Keep
├── ode1.py, ode2.py, ...      # Not relevant
├── save/                       # Old results
└── ... (many more files)
```

### Refactored Structure (Clean)
```
spinn_paper/
├── src/                        # Clean, organized source
│   ├── core.py                # Unified base classes
│   ├── pde.py                 # PDE utilities
│   ├── neural_network.py      # SPINN implementation
│   └── cavity_pde.py          # Main paper implementation
├── examples/                   # Clear examples
├── tests/                     # Proper testing
└── documentation              # Good documentation
```

## Benefits of Refactoring

1. **Maintainability**: Clear separation of concerns
2. **Reusability**: Modular design allows easy extension
3. **Documentation**: Proper README and examples
4. **Testing**: Unit tests for validation
5. **Packaging**: Proper Python package structure
6. **Paper Fidelity**: Preserves all original functionality

## Usage

### Quick Start
```bash
cd spinn_paper
pip install -r requirements.txt
python examples/simple_example.py
```

### Full Paper Reproduction
```bash
python examples/cavity_example.py
```

### Run Tests
```bash
python tests/test_cavity.py
```

## Migration Guide

If you were using the original code:

1. **Main implementation**: `cavity_lubin_slip.py` → `src/cavity_pde.py`
2. **Common utilities**: `common_nomayavi.py` → `src/core.py`
3. **Neural network**: `spinn2d_nomayavi.py` → `src/neural_network.py`
4. **PDE base**: `pde2d_base_nomayavi.py` → `src/pde.py`

The refactored code maintains the same functionality but with a cleaner, more maintainable structure.
