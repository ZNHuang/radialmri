import sys
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy.io import loadmat
from simulation_and_reconstruction import *
#from mrishow import *
import argparse
plt.rcParams.update({'figure.max_open_warning': 0})

### Note, not functioning due to missing of one file which I will add ASAP
def main():
    #---input---
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type = str)
    parser.add_argument("-o", "--output", type = str)
    parser.add_argument("-n", "--niteration", type = int, default = 32)
    parser.add_argument("-s", "--nspokes", type = int, default = 21)
    args = parser.parse_args()
    filetoload = args.input
    outputname = args.output
    niteration = args.niteration

    #---load---
    loadedsim = loadmat(filetoload)

    #---reconstruction parameters---
    im_size = (320, 320)
    resolution = 320 #
    grid_size = (640, 640) #for gridding of the nufft
    niter = 160 #32000#
    inp_spokes = args.nspokes# each time point has 21 spokes
    nt = 22 #number of time points
    nspokes_sim =  nt * inp_spokes #630 #30*21 = 630
    spokelength = 640
    nc = 16

    loadedsim = loadmat(filetoload)
    simulated_kspace_21 = torch.tensor(loadedsim['simulated_kspace_21'])
    smap_loaded = numpy2torch(loadedsim['smap_complex'])
    smap_loaded = smap_loaded.permute(1,0,2,3).unsqueeze(0)
    traj_21 = torch.tensor(loadedsim['traj'])
    dcomp_21 = torch.tensor(loadedsim['dcomp'])
    noise = torch.tensor(loadedsim['noise']).to(device, dtype)
    simulated_target = loadedsim['simulated_target']
    target_recombine = loadedsim['target_recombine']
    masksloaded = loadedsim['mask']
    nl = loadedsim['nl'][0][0]

    #---variables process---
    simulated_kspace_21_np = simulated_kspace_21.view(
            22, 16, 2, inp_spokes, 640).cpu().numpy()

    simulated_kspace_21_np = np.concatenate(
            [simulated_kspace_21_np[i] for i in range(nt)], axis=2)

    smap_reest = cal_coil_sensitivities(simulated_kspace_21_np,
                       get_traj(nt*inp_spokes, 640),
                       resolution=320).unsqueeze(0)

    smap_reest_nm = normalized_coilsmap(smap_reest)
    smap_nm = normalized_coilsmap(smap_loaded)

    target_recombine = numpy2torch(loadedsim['target_recombine'], device).\
            permute(1, 0, 2, 3)

    #print(target_recombine.shape)
    target_recombine_np = target_recombine.cpu().numpy()

    #---model class initialization---
    Rdmodel = RadialModel(grid_size=grid_size, im_size= im_size).\
            to(device, dtype)

    x_adjn = Rdmodel.adjoint(
            y = (simulated_kspace_21*torch.sqrt(dcomp_21)).to(device, dtype),
            k = traj_21.to(device, dtype),
            coil_sensitivities= smap_reest_nm.to(device, dtype),
            #coil_sensitivities= smap_nm.to(device, dtype),
            w = torch.sqrt(dcomp_21).to(device, dtype))
    #to make a one matrix for dcomp

    #---input: adjoint nufft and save---
    #print(x_adjn.shape)
    x_adjn_np = torch2numpy(x_adjn, complexdim=1)
    #print(x_adjn_np.shape)

    torch.save(x_adjn, '{}_adjnufft.pt'.format(outputname))

    #---reconstruction---
    if niteration <= 0: #only get the adjoint nufft result
        return

    # 2.5% of maximum adjnufft signal
    lambda1 = 2.5e-2 * max(np.abs(x_adjn_np.flatten()))
    #print(lambda1)

    dcomp_trivial = torch.ones(dcomp_21.shape).to(device)

    x_cgrecon_val2 = RadialRecon(
            kspace=((simulated_kspace_21).to(device, dtype)+noise),
            traj = traj_21.to(device, dtype),
            #coil_sensitivities = smap.to(device, dtype),
            #coil_sensitivities = smap_nm.to(device, dtype),
            coil_sensitivities = smap_reest_nm.to(device, dtype),
            w = dcomp_trivial.to(device, dtype),
            grid_size = grid_size,
            im_size = im_size,
            tolerance = 1e-6,

            lambda1 = lambda1,
            device = device,
            dtype = dtype,
            keep_history = True,
            niter = niteration)

    temp = x_cgrecon_val2[0][:,0,...]+1j*x_cgrecon_val2[0][:,1,...]
    torch.save(temp, '{}.pt'.format(outputname))

    save_gif(list_img=[np.abs(temp[i]) for i in range(22)],
            fname = 'iGRASP_{}.gif'.format(outputname))

if __name__ == "__main__":
    main()
