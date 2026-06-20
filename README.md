# 🔬 Hydromatic Simulator

Hydromatic Simulator provides a portable virtual space to configure and test the dynamic shape deformation of Soft Hydromatic Actuators (SHA) based on their binary-encoded structural designs. 

The program contains an implementable nodal position generator models presented in the paper "AI-Driven Hydromatic Actuators for Biomimetic Motion." 

Based on the numerical simulation data from 10400 binary-encoded designs of SHAs, the 16 generator models simultaneously predict the dynamic nodal trajectories on XY-plane over the 24 h deformation period until equilibrium. The trajectory data were assembled to draw the 16-coordinate shape deformation results. The input SHA structural design can be manually configured through GUI editor. 

More detailed description of the code's functionality can be found in the Supporting Information of the manuscript.

![Static Badge](https://img.shields.io/badge/DOI-10.5281/zenodo.19259307-blue?link=!%5BStatic%20Badge%5D(https%3A%2F%2Fimg.shields.io%2Fbadge%2FDOI-https%253A%252F%252Fdoi.org%252F10.1073%252Fpnas.2424405122-blue))


## 1. System Requirements
### Software dependencies and operating systems
- Operating Systems: Windows 10/11, macOS (12.0+).
- Programming Language: Python 3.8.10 or higher.
- Dependencies:
    - Pillow ≥ 10.0.0
    - Matplotlib ≥ 3.8.0
    - TensorFlow ≥ 2.15.0
    - Keras ≥ 3.0.0
    - NumPy ≥ 1.26.0
    - pandas ≥ 2.1.0
    - scikit-learn ≥ 1.4.0

## 2. Installation Guide

- Estimated software install time: 3-5 minutes on a normal desktop computer.

## Installation (Mac)

1. Clone the repository and locate to the folder:
```
git clone https://github.com/bae-research-group/Hydromatic_Simulator.git
cd Hydromatic_Simulator
```
2. Set up a virtual environment (Recommended):
```
python3 -m venv venv
source venv/bin/activate
```
3. Install the dependencies:
```
pip install -r requirements.txt
```
4. Set up Git LFS (to download the large (PNG) files):
```
git lfs install
git lfs pull
```
5. Run from source:
```
python main.py
```
or install as a CLI app:
```
pip install .
Hydromatic_Simulator
```

## Installation (Windows)

1. Clone the repository and locate to the folder:
```
git clone https://github.com/bae-research-group/Hydromatic_Simulator.git
cd Hydromatic_Simulator
```
2. Set up a virtual environment (Recommended):
```
python -m venv venv
venv\Scripts\activate
```
3. Install the dependencies:
```
pip install -r requirements.txt
```
4. Set up Git LFS (to download the large (PNG) files):
```
git lfs install
git lfs pull
```
5. Run from source:
```
python main.py
```
or install as a CLI app:
```
pip install .
Hydromatic_Simulator
```

## 3. Demo
The detailed software demo and the instructions for use are available in Supporting Video S7 of the manuscript.
- Estimated runtime to reproduce the demonstration shown in Supporting Video S7: 1-2 minutes on a normal desktop computer.

## Acknowledgements
This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Ministry of Science and ICT (MSIT), RS-2025-00557115.

## License
This project is licensed under the MIT License. (See the LICENSE file for details).

## Open Source Repository
The code is available at: https://github.com/bae-research-group/Hydromatic_Simulator.git (DOI: 10.5281/zenodo.19259307).

## Citing
```bibtex

@article{
}
```
