# Radial MRI $k$-space Simulation and Reconstruction

This repository/package simulate radial kspace from a series of MR images. By 
using GPU + pytorch + [torchkbnufft](https://github.com/mmuckley/torchkbnufft) we 
are able to make the simulation and reconstruct faster, therefore suitable 
for machine learning application. You may also use the code just for 
[iGRASP](https://pubmed.ncbi.nlm.nih.gov/24142845/) or [CG-SENSE](https://onlinelibrary.wiley.com/doi/10.1002/mrm.1241) reconstruction.

![Diagram](/fig1b.png)

## Getting Started

The code written in this repository/package requires:

| First Header  | Second Header |
| ------------- | ------------- |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |


|Tool     |Version      |
| ------- | ----------- |
|Python   | **3.6.8**   |
| ------- | ----------- |
|torch    | **1.1.0**   |
| ------- | ----------- |
|torchkbnufft| **0.2.1**|

The specific version are very important.(Other Python3 versions might be OK) Later version of the software are tested, and will result in either GPU memory inefficiency or complete failure of running the code. 

To verify the version of the python, in shell type 

```
python --version
```

In python, verify the version of the torch by:

```
import torch
torch.__version__
```

To Install the package with a specific version: ```pip install 'PackageName==1.1.0'```

If running into problem about environment set-up. Contact author for assistant. A Singularity container is set up for this by the author to run these scripts. The author will be happy to share that container if it helps.

## Switch Between GPU and CPU Computing

This package can be used with or without GPU. GPU computing is faster but can't handle larget image samples. In [radialmri/simulation_and_reconstruction.py](/radialmri/simulation_and_reconstruction.py) file, switch between annotation status of line 21 and line 22 allow you to switch between GPU and CPU computing. 

## Detailed Usage

Please refer to the jupyter notebooks in the test folder. test.ipynb shows the basics and Example_Crane.ipynb shows the details. 

## Testing Data

Due to patient data involved, I'm not able to push realistic example as testing data to the repository. In the test/ directory, there is an example based on a short vedio for demonstration purpose. Unrealistic coil sensitivites and phase were used for simplicity. This example will show how the package works from simulation to reconstruction.


## Reference
1. Feng L, Grimm R, Block KT, Chandarana H, Kim S, Xu J, Axel L, Sodickson DK, Otazo R. Golden-angle radial sparse parallel MRI: combination of compressed sensing, parallel imaging, and golden-angle radial sampling for fast and flexible dynamic volumetric MRI. Magn Reson Med. 2014 Sep;72(3):707-17. doi: 10.1002/mrm.24980. Epub 2013 Oct 18. PMID: 24142845; PMCID: PMC3991777.
2. Pruessmann KP, Weiger M, Börnert P, Boesiger P. Advances in sensitivity encoding with arbitrary k-space trajectories. Magn Reson Med. 2001 Oct;46(4):638-51. doi: 10.1002/mrm.1241. PMID: 11590639.
3. {Muckley2019,
  author = {Muckley, M.J. et al.},
  title = {Torch KB-NUFFT},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/mmuckley/torchkbnufft}}
}
