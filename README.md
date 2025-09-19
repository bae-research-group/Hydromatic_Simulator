# 🔬 Hydromatic Simulator

Hydromatic Simulator provides a portable virtual space to configure and test the dynamic shape deformation of Soft Hydromatic Actuators (SHA) based on their binary-encoded structural designs. 

The program contains an implementable nodal position generator models presented in the paper "AI-Driven Hydromatic Actuators for Biomimetic Motion." 

Based on the numerical simulation data from 10400 binary-encoded designs of SHAs, the 16 generator models simultaneously predict the dynamic nodal trajectories on XY-plane over the 24 h deformation period until equilibrium. The trajectory data were assembled to draw the 16-coordinate shape deformation results. The input SHA structural design can be manually configured through GUI editor. 

![Static Badge](https://img.shields.io/badge/DOI-10.1073%2Fpnas.2424405122-blue?link=!%5BStatic%20Badge%5D(https%3A%2F%2Fimg.shields.io%2Fbadge%2FDOI-https%253A%252F%252Fdoi.org%252F10.1073%252Fpnas.2424405122-blue))

## Installation

1. Clone the repository:
```
git clone https://github.com/bae-research-group/Hydromatic_Simulator.git
cd Hydromatic_Simulator
```
2. Install the dependencies:
```
pip install -r requirements.txt
```
3. (Optional) Set up Git LFS:
```
git lfs install
git lfs pull
```
4. Run from source:
```
python main.py
```
or install as a CLI app:
```
pip install .
Hydromatic_Simulator
```

## Acknowledgements

-.


## Citing
```bibtex

@article{
}
```
