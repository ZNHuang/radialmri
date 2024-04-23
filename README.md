# Radial(or Cartesian) MR <i>k</i>-space Simulation and Image Reconstruction

This repository/package simulate radial kspace from a series of MR images. By 
using GPU/CPU + pytorch + [torchkbnufft](https://github.com/mmuckley/torchkbnufft) we 
are able to make the simulation and reconstruction faster, therefore suitable 
for machine learning applications. You may also use the code just for 
[iGRASP](https://pubmed.ncbi.nlm.nih.gov/24142845/)(momentarily unavailable) or [CG-SENSE](https://onlinelibrary.wiley.com/doi/10.1002/mrm.1241) reconstruction, Radial or Cartesian.

![Diagram](/fig1b.png)

## Getting Started

The code written in this repository/package requires:

| Tool    | Version     |
| ------- | ----------- |
|Python   | **3.6.8**   |
|torch    | **1.1.0**   |
|torchkbnufft| **0.2.1**|

The specific version are very important. Later version of the torch and torchkbnufft are tested (Other Python3 versions might be OK, for instance Python 3.8.3), and will result in either GPU memory inefficiency or complete failure of running the code. 

To verify the version of the python, in shell type 

```
python --version
```

In python, verify the version of the torch by:

```
import torch
torch.__version__
```

To install the package with a specific version: ```pip install 'PackageName==1.1.0'```. 

You may try to use the requirement file by ```pip install -r requirements.txt```, however I don't find it bug-free since I saw some issue after generating the requirement.txt using ```pip freeze``` and use it on another computer. You can ignore it. I would recommend pip install everything in a conda environment and be sure to use the torch and torchkbbufft version as specified above and let conda resolve the other dependency versions.

If running into problem about environment set-up. Contact author for assistant. ~~A Singularity container is set up for this by the author to run these scripts.~~ It was not too difficult to set up conda environment myself again after some time away from this project and losing the environment set up.

## Switch Between GPU and CPU Computing

This package can be used with or without GPU. GPU computing is faster but CPU can handle larger image samples. In [radialmri/simulation_and_reconstruction.py](/radialmri/simulation_and_reconstruction.py#L21) file, switch between annotation status of line 21 and line 22 allow you to switch between GPU and CPU computing. 

Similar operation should be done to switch for [complex_operations.py](https://github.com/ZNHuang/radialmri/blob/main/radialmri/complex_operations.py#L3) 

Note in this latest version, I switched all code to run on CPU as it does not require advanced hardware. But GPU computing is still working. The ```Example-Crane.ipynb``` requires some RAM to work (20 - 30 GiB) as it is now. But you can dramatically reduce it by using less frames. 

## Detailed Usage

Please refer to the jupyter notebooks in the example folder. test.ipynb shows the basics and [Example_Crane.ipynb](/example/Example-Crane.ipynb) shows the details. 

## Testing Data

Due to patient data involved, I'm not able to push realistic example as testing data to the repository. In the [example](/example/) directory, there is an example based on a short vedio for demonstration purpose. Unrealistic coil sensitivites and phase were used for simplicity. This example will show how the package works from simulation to reconstruction.

## Next Step
A lot of refactoring and clean up need to be done for this repository. 

## Reference
1. Feng L, Grimm R, Block KT, Chandarana H, Kim S, Xu J, Axel L, Sodickson DK, Otazo R. Golden-angle radial sparse parallel MRI: combination of compressed sensing, parallel imaging, and golden-angle radial sampling for fast and flexible dynamic volumetric MRI. Magn Reson Med. 2014 Sep;72(3):707-17. doi: 10.1002/mrm.24980. Epub 2013 Oct 18. PMID: 24142845; PMCID: PMC3991777.
2. Pruessmann KP, Weiger M, Börnert P, Boesiger P. Advances in sensitivity encoding with arbitrary k-space trajectories. Magn Reson Med. 2001 Oct;46(4):638-51. doi: 10.1002/mrm.1241. PMID: 11590639.
3. {Muckley2019,
  author = {Muckley, M.J. et al.},
  title = {Torch KB-NUFFT},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/mmuckley/torchkbnufft
	}}
}
