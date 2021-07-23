import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import data
import time
import h5py
from scipy.ndimage import gaussian_filter

from torchkbnufft import KbNufft as TorchKbNufft
from torchkbnufft import AdjKbNufft as TorchAdjKbNufft
from torchkbnufft import MriSenseNufft, AdjMriSenseNufft

from scipy.io import loadmat, savemat
import argparse
import warnings
from simulation_and_reconstruction import *

plt.rcParams.update({'figure.max_open_warning': 0})

dtype = torch.float
device = torch.device('cuda')
#device = torch.device('cpu')

im_size = (320, 320)
resolution = 320 #
grid_size = (640, 640) #for gridding of the nufft
spokelength = 640
nc = 16

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--optimizer', help='Optimizer for reconstruction', default='GD')
parser.add_argument('-c', '--caseidx', help='Grount truth case ID for simulation', default='CASEID')
parser.add_argument('-s', '--smap', help='coil sensitivities index', default=None)
parser.add_argument('-n', '--noiselevel', help='std of simulated noise added to simulated kspace', default=None)
parser.add_argument('-l', '--lambda1', help='temporal regularization parameter', default=None)
parser.add_argument('-d', '--dir', help='directory to save', default='.')
parser.add_argument('-f', '--file', help='simulated data file to load when only runnig reconstruction', default = None)
parser.add_argument('--history', help='Save optimization history or not, default False', action='store_true', default=False)
parser.add_argument('--spokes', help='spokes per frame/time point', default=21)
parser.add_argument('-t', '--nt', help='number of time point', default=22)
parser.add_argument('--Cartesian', help = 'Run cartesian simulation and save the fully sampled kspace', action = 'store_true', default = False)

args = parser.parse_args()
caseidx = str(args.caseidx)
savedir = str(args.dir)
#onlyrecon = bool(args.onlyrecon)
filetoload = str(args.file)
ifhistory = bool(args.history)
inp_spokes = int(args.spokes) #default is 21
nt = int(args.nt)#default is 22
Cartesian = bool(args.Cartesian) #Mon Jun  7 20:01:02 EDT 2021

nspokes_sim =  nt*inp_spokes #630 #30*21 = 630
opt = str(args.optimizer)

if args.noiselevel is None:
    nl = None
else:
    nl = float(args.noiselevel)

if args.smap is None:
    csidx = None
else:
    csidx = str(args.smap)

if args.lambda1 is not None:
    lambda1 = float(args.lambda1)
else:
    lambda1 = None

print('input noise level = {}'.format(nl))

#from io import StringIO

def main():
    global nl
    global lambda1
    print('Running simulation')
    #------------------load and preprocess---------------------
    datadir = '../sample/sim_{}.mat'.format(caseidx)
    dataload = loadmat(datadir)
    #print(dataload.keys())

    masksloaded = dataload['mask']

    simulated_image = dataload['simImg']
    simulated_image = np.flipud(np.array(simulated_image))#flip up and down

    simulated_target = np.array(simulated_image, dtype='complex128')
    simulated_target = simulated_target.swapaxes(0, 2).swapaxes(1, 2)
    #print(simulated_target.shape, simulated_target.dtype)

    target = np.stack((simulated_target.real, simulated_target.imag), axis=1)
    target = torch.tensor(target, dtype=dtype, device=device)
    #print(target.shape)

    #---------------load simulated coil sensitivities-----------------
    if csidx is not None:
        cs_loaded = loadmat('/gpfs/home/zh1115/knolllabspace/hzn/storage/simulate_coil_sensitivities/coil_sensitivities/cs_{}.mat'.format(csidx))#input directory of the simulated coil sensitivities
        #csidx = '1'

        #print(cs_loaded.keys())
        cs_loaded['imgSens'].dtype
        smap_loaded = numpy2torch(cs_loaded['imgSens'], device =device)
        smap_loaded = smap_loaded.permute(3,0,1,2).unsqueeze(0)
        smap_complex = cs_loaded['imgSens'].swapaxes(0, 2).swapaxes(1, 2)
    else:
        smap_complex = dataload['smap_complex']
        smap_loaded = numpy2torch(smap_complex, device =device)
        smap_loaded = smap_loaded.permute(1,0,2,3).unsqueeze(0)

    #print('smap_loaded.shape', smap_loaded.shape)
    #print(smap_complex.shape)
    temporal_coilimg = np.array([sim_coil(simulated_target[i,:,:], smap_complex, coild=0) for i in range(nt)])
    #print(temporal_coilimg.shape)
    target_recombine= np.array([temporal_coilimg[:,i]*np.conj(smap_complex[i,:,:]) for i in range(temporal_coilimg.shape[1])])
    target_recombine= np.sum(target_recombine, axis=0)

    #---------------simulation---------------------
    if Cartesian:
        print("target.shape", target.shape)
        smap_loaded_cartesian = smap_loaded.repeat(nt, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        simulated_kspace_cartesian, recon_fromsim_cartesian = CartesianSimulation(\
                target = target,
                smap = smap_loaded_cartesian)
        simulated_kspace_cartesian = simulated_kspace_cartesian.permute(1, 0, 4, 2, 3)
        print('simulated_kspace_cartesian.shape, recon_fromsim_cartesian.shape', simulated_kspace_cartesian.shape, recon_fromsim_cartesian.shape)
        #recon_fromsim_21 = torch2numpy(recon_fromsim_21.permute(0,2,3,1))
        #plt.figure(figsize=(10, 10))
        #plt.imshow(np.abs(recon_fromsim_21[10]))
        #plt.show()

        np.random.seed(0)
        if nl is None:
            #determine the noise level if it's not provided by 1/50 of the kspace center
            nl = extract_kspacecenter_cartesian(simulated_kspace_cartesian, nt = nt, nc = nc)/100# Thu Oct 22 12:46:49 EDT 2020  #TODO
            print('noise level', nl)

        noise = nl*torch.tensor(np.random.normal(size =simulated_kspace_cartesian.shape)).to(device, dtype)

        #--------------save---------------
        savematname = '{}/{}_cartesian_origsmap_{}_{:.2e}.mat'.format(savedir, caseidx, csidx, nl)
        savemat(savematname, mdict ={
                'simulated_kspace_cartesian': simulated_kspace_cartesian.cpu().numpy(),
                'simulated_target': simulated_target,
                'target_recombine': target_recombine,
                'mask': masksloaded,
                'smap_complex': smap_complex,
                'noise': noise.cpu().numpy(),
                'nl': nl
            })

        target_gifname = '{}/target_case{}_cs{}.gif'.format(savedir, caseidx, csidx)
        save_gif(simulated_target, target_gifname)

        target2_gifname = '{}/targetrecombine_case{}_cs{}.gif'.format(savedir, caseidx, csidx)
        save_gif(target_recombine, target2_gifname)
        return

    #------Radial Simulation------
    print("target.shape", target.shape, target.dtype,\
          "smap.shape", smap_loaded.shape, smap_loaded.dtype)

    simulated_kspace_21, recon_fromsim_21, traj_21, dcomp_21 \
    = RadialSimulation(target = target,
                       spokespertime=inp_spokes,
                       nt=nt, nc=nc,
                       spokelength=spokelength,
                       smap=smap_loaded,
                       grid_size = grid_size,
                       im_size= im_size)

    #print("Output dcomp.shape", dcomp_21.shape, dcomp_21.dtype)
    recon_fromsim_21 = torch2numpy(recon_fromsim_21.permute(0,2,3,1))

    #plt.figure(figsize=(10, 10))
    #plt.imshow(np.abs(recon_fromsim_21[10]))
    #plt.show()

    np.random.seed(0)
    if nl is None:
        #determine the noise level if it's not provided by 1/50 of the kspace center
        #nl = extract_kspacecenter(simulated_kspace_21, spokelength=640)/50
        nl = extract_kspacecenter(simulated_kspace_21, spokelength=640)/100
        print('noise level', nl)

    noise = nl*torch.tensor(np.random.normal(size =simulated_kspace_21.shape))\
            .to(device, dtype)

    #--------------save---------------
    savematname = '{}/{}_sim21spokes_origsmap_{}_{:.2e}.mat'.\
    format(savedir, caseidx, csidx, nl)

    savemat(savematname, mdict ={
            'simulated_kspace_21': simulated_kspace_21.cpu().numpy(),
            'simulated_target': simulated_target,
            'target_recombine': target_recombine,
            'mask': masksloaded,
            'smap_complex': smap_complex,
            'noise': noise.cpu().numpy(),
            'traj': traj_21.cpu().numpy(),
            'dcomp': dcomp_21.cpu().numpy(),
            'nl': nl})

    target_gifname = '{}/target_case{}_cs{}_Recon_iGRASP.gif'.\
    format(savedir, caseidx, csidx)

    save_gif(simulated_target, target_gifname)
    target2_gifname = '{}/targetrecombine_case{}_cs{}_Recon_iGRASP.gif'.\
    format(savedir, caseidx, csidx)

    save_gif(target_recombine, target2_gifname)

if __name__ == "__main__":
    main()
