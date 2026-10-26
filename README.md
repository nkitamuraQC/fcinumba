# fcinumba
A LLVM experimental code for pythonic FCI

## Dependency
- python: 3.10.10
- pyscf: 2.13.0
- numba: 0.66.0

## Usage
- see test.py

## Installation
Run the following command in the directory where `pyproject.toml` is located
```sh
pip install -e .
```

## Results
The σ-vector calculation, which is the computational bottleneck in FCI calculations, was JIT-compiled with LLVM using Numba. The implementation was based on the FCI code provided in PySCF, which is written in Python.
For small systems such as H₂, the LLVM-FCI calculation took longer than the original Python-based FCI, likely because of compilation and other overheads. However, as the number of hydrogen atoms increased, the performance gap (ratio) narrowed; for a 12-atom hydrogen system, the difference in calculation time was small. This suggests that LLVM-FCI may exhibit better scaling for larger systems.
We also found that using numpy.dot was faster than explicitly writing the corresponding operations as loops, particularly in parts originally expressed with einsum.

## License

This repository is distributed under the MIT License, except for files derived from PySCF, which retain their applicable Apache License, Version 2.0 terms.

PySCF-derived files retain their original copyright and author notices. See `LICENSE-APACHE-2.0` for the Apache License, Version 2.0, and `NOTICE-PySCF` for applicable attribution notices.

Modifications to PySCF-derived files are identified in the corresponding source-file headers.
