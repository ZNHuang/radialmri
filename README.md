# Radial MRI Kspace Simulation

This repository/package simulate radial kspace from a series of MR images.

![Diagram](/fig1b.png)

## Getting Started

The code written in this repository/package requires

Python **3.6.8**

torch **1.1.0**

torchkbnufft **0.2.1**

for computation. The specific version are very important. Later version of the software are tested, and will result in either GPU memory inefficiency or complete failure of running the code. 

To verify the version of the python, in shell type 

'''
python --version
'''

In python, verify the version of the torch by:

'''
import torch
torch.__version__
'''

To Install the package with a specific version: pip install 'PackageName==1.1.0'.

If running into problem about environment set-up. Contact author for assistant. A Singularity container is set up for this by the author to run these scripts. The author will be happy to share that container if it helps.

## Switch between GPU and CPU computing for simulation and (i)GRASP reconstruction

In [radialmri/simulation_and_reconstruction.py](/radialmri/simulation_and_reconstruction.py) file, switch between annotation status of line 21 and line 22.

## For Detailed Usage, please refer to the jupyter notebooks in the test folder

## Testing Data

Due to patient data involved, I'm not able to push the exmaple testing data to the repository. I'm working on an example based on a short vedio for demonstration purpose.
