"""
Script to identify and clean up duplicate files from the original codebase.
This script helps identify which files can be safely removed.
"""

import os
import shutil
from pathlib import Path


def identify_duplicates():
    """Identify duplicate files in the original codebase."""
    
    # Files that are duplicates or can be removed
    duplicates_to_remove = [
        # Duplicate common files
        "code/common.py",  # Keep common_nomayavi.py as it has cuda.empty_cache()
        
        # Duplicate cavity implementations (keep only cavity_lubin_slip.py)
        "code/cavity.py",
        "code/cavity_pmd_lubin.py", 
        "code/cavity_pmd_lubin_nomayavi.py",
        "code/cavity_pmd_lubin_nomayavi_one.py",
        "code/cavity_pmd_lubin_nomayavi_trans2m.py",
        "code/cavity_pmd_lubin_nomayavi_trans2m_dimension.py",
        "code/cavity_lubin_slip_threemag.py",
        
        # Duplicate spinn implementations
        "code/spinn2d.py",  # Keep spinn2d_nomayavi.py
        
        # Duplicate pde base files
        "code/pde2d_base.py",  # Keep pde2d_base_nomayavi.py
        
        # Old test files that are not needed
        "code/ode1.py",
        "code/ode2.py", 
        "code/ode3.py",
        "code/ode4.py",
        "code/ode1_var.py",
        "code/ode3_var.py",
        "code/ode4_pinn.py",
        "code/ode1_fourier.py",
        "code/ode3_fourier.py",
        "code/ode4_fourier.py",
        
        # Old 1D implementations not relevant to paper
        "code/advection1d.py",
        "code/advection1d_fd_spinn.py",
        "code/burgers1d.py",
        "code/burgers1d_fd_spinn.py",
        "code/heat1d.py",
        "code/heat1d_fd_spinn.py",
        "code/allen_cahn_fd.py",
        "code/ode_base.py",
        "code/ibvp2d_base.py",
        "code/var1d_base.py",
        "code/fd_spinn1d_base.py",
        "code/fourier1d_base.py",
        "code/pinn1d_base.py",
        "code/spinn1d.py",
        
        # Poisson examples not relevant to main paper
        "code/poisson2d_sine.py",
        "code/poisson2d_pulse.py", 
        "code/poisson2d_irreg_dom.py",
        "code/poisson2d_square_slit.py",
        
        # Miscellaneous files
        "code/plotmagnet.py",
        "code/gconv2d.py",
        "code/B.csv",
        
        # Old save directories
        "code/save/",
        "code/D_/",
        "code/fem/",
        "code/mesh_data/",
        "code/misc/",
    ]
    
    return duplicates_to_remove


def create_backup():
    """Create a backup of the original code."""
    print("Creating backup of original code...")
    
    # Create backup directory
    backup_dir = Path("code_backup")
    backup_dir.mkdir(exist_ok=True)
    
    # Copy original code directory
    if Path("code").exists():
        shutil.copytree("code", backup_dir / "code", dirs_exist_ok=True)
        print(f"✓ Backup created at {backup_dir}")
    else:
        print("✗ Original code directory not found")


def clean_duplicates():
    """Clean up duplicate files."""
    print("Cleaning up duplicate files...")
    
    duplicates = identify_duplicates()
    
    for file_path in duplicates:
        full_path = Path(file_path)
        if full_path.exists():
            if full_path.is_file():
                full_path.unlink()
                print(f"✓ Removed file: {file_path}")
            elif full_path.is_dir():
                shutil.rmtree(full_path)
                print(f"✓ Removed directory: {file_path}")
        else:
            print(f"- File not found: {file_path}")
    
    print("✓ Cleanup completed")


def main():
    """Main cleanup function."""
    print("SPINN Project Cleanup")
    print("=" * 30)
    
    # Create backup first
    create_backup()
    
    # Clean duplicates
    clean_duplicates()
    
    print("\n✓ Cleanup completed successfully!")
    print("The original code has been backed up and duplicates removed.")
    print("The new clean structure is in the 'spinn_paper' directory.")


if __name__ == "__main__":
    main()
