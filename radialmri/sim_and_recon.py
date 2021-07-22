#Changed from sim_and_iGRASP_intensitycorrection.py by change the from simulation_and_reconstruction_IC to simulation_and_reconstruction_beta. Will be able to generate simulation with different #spokes according to the input
import sys
sys.path = ['', '/gpfs/share/apps/anaconda3/gpu/5.2.0/lib/python3.6/site-packages', '/cm/local/apps/cuda/libs/current/pynvml', '/gpfs/share/apps/intel/2019/advisor_2019.1.0.579143/pythonapi', '/gpfs/share/apps/anaconda3/gpu/5.2.0/lib/python3.6/site-packages/openmmlib-0.0.0-py3.6-linux-x86_64.egg', '/gpfs/home/zh1115/knolllabspace/envs/pytorch4vn/lib/python36.zip', '/gpfs/home/zh1115/knolllabspace/envs/pytorch4vn/lib/python3.6', '/gpfs/home/zh1115/knolllabspace/envs/pytorch4vn/lib/python3.6/lib-dynload', '/gpfs/home/zh1115/.local/lib/python3.6/site-packages', '/gpfs/home/zh1115/knolllabspace/envs/pytorch4vn/lib/python3.6/site-packages', '/gpfs/home/zh1115/knolllabspace/envs/pytorch4vn/lib/python3.6/site-packages/optoth-0.2.0-py3.6-linux-x86_64.egg']\
+sys.path #Not needed if all modules can be imported properly

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import data
#from scipy.ndimage.morphology import binary_fill_holes
#import pydicom
#import matplotlib.pyplot as plt
import time
import h5py
from scipy.ndimage import gaussian_filter

from torchkbnufft import KbNufft as TorchKbNufft
from torchkbnufft import AdjKbNufft as TorchAdjKbNufft
from torchkbnufft import MriSenseNufft, AdjMriSenseNufft
#sys.path.append("../")
#from mri.dcomp_calc import calculate_radial_dcomp_pytorch
#sys.path.append("/gpfs/home/zh1115/fast_mri2/grasp/")
#from optimizer.quadraticcg import QuadraticCg  # noqa: E402

from scipy.io import loadmat, savemat
import argparse
import warnings
#sys.path.append('/gpfs/home/rns365/fast_mri/')
#from experimental.pmin.grasp import grasp_recon

dtype = torch.float
device = torch.device('cuda')
#device = torch.device('cpu')

im_size = (320, 320)
resolution = 320 #
grid_size = (640, 640) #for gridding of the nufft
niter = 160#32000#
spokelength = 640
nc = 16

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--optimizer', help='Optimizer for reconstruction', default='GD')
parser.add_argument('-r', '--onlyrecon', help='Only running reconstruction, simulated data required', action='store_true')
parser.add_argument('-c', '--caseidx', help='Grount truth case ID for simulation', default='CASEID')
parser.add_argument('-s', '--smap', help='coil sensitivities index', default=None)
parser.add_argument('-n', '--noiselevel', help='std of simulated noise added to simulated kspace', default=None)
parser.add_argument('-i', '--iterations', help='iterations to run', default=0)
parser.add_argument('-l', '--lambda1', help='temporal regularization parameter', default=None)
parser.add_argument('-d', '--dir', help='directory to save', default='.')
parser.add_argument('-f', '--file', help='simulated data file to load when only runnig reconstruction', default = None)
parser.add_argument('--history', help='Save optimization history or not, default False', action='store_true', default=False)
parser.add_argument('--spokes', help='spokes per frame/time point', default=21)
parser.add_argument('-t', '--nt', help='number of time point', default=22)
parser.add_argument('--Cartesian', help = 'Run cartesian simulation and save the fully sampled kspace', action = 'store_true', default = False)

args = parser.parse_args()
niter = int(args.iterations)
caseidx = str(args.caseidx)
savedir = str(args.dir)
onlyrecon = bool(args.onlyrecon)
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

print('input noise level = {}, iterations = {}'.format(nl, niter))

if niter <= 0:
    warnings.warn('Iteration <= 0, no reconstruction, only simulation')
else:
    print('Will run iGRASP(if lambda1 != None) reconstruction for {} iterations with lamdba1={}'.format(niter, lambda1))

from simulation_and_reconstruction import *

# catch the loss
from io import StringIO
import sys

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

def main():
    global nl
    global lambda1
    if not onlyrecon:
        print('Running simulation')
        #------------------load and preprocess---------------------
        #maskdir = '/gpfs/home/zh1115/knolllabspace/hzn/storage/newsimulation/mask07_processed.mat'

        #datadir = '/gpfs/home/zh1115/knolllabspace/hzn/storage/newsimulation/simulated_20200630/sim_{}.mat'.format(caseidx) #simulated images
        #datadir = '/gpfs/home/zh1115/knolllabspace/hzn/storage/newsimulation/simulated_20200811/sim_{}.mat'.format(caseidx) #simulated images #Mon Oct 12 15:43:22 EDT 2020
        datadir = '/gpfs/home/zh1115/knolllabspace/hzn/storage/newsimulation/simulated_20210128/sim_{}.mat'.format(caseidx) #simulated images #Thu Jan 28 14:14:24 EST 2021
        #datadir = '../sample/sim_{}.mat'.format(caseidx) #simulated images #Thu Jan 28 14:14:24 EST 2021
        #caseidx = 'BC18'

        dataload = loadmat(datadir)
        print(dataload.keys())

        #masksloaded = loadmat(maskdir)
        #masksloaded = np.array(masksloaded['mask'])
        masksloaded = dataload['mask']
        #print(masksloaded.dtype, masksloaded['background'][0,0].shape)

        #imagedir = '/gpfs/home/zh1115/knolllabspace/hzn/storage/newsimulation/sim_pop_complexbaseimage.mat'

        #simulated_image = h5py.File(imagedir, 'r')
        #simulated_image = loadmat(imagedir)
        simulated_image = dataload['simImg']
        #simulated_image = simulated_image['sim_pop']
        simulated_image = np.flipud(np.array(simulated_image))#flip up and down
        #print(simulated_image.shape, simulated_image.dtype)

        #plt.imshow(masksloaded['glandular_tissue'][0][0])
        #plt.show()

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

        print('smap_loaded.shape', smap_loaded.shape)
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
                print('nl', nl)

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

            target_gifname = '{}/target_case{}_cs{}_Recon_iGRASP.gif'.format(savedir, caseidx, csidx)
            save_gif(simulated_target, target_gifname)

            target2_gifname = '{}/targetrecombine_case{}_cs{}_Recon_iGRASP.gif'.format(savedir, caseidx, csidx)
            save_gif(target_recombine, target2_gifname)

            return


        print("target.shape", target.shape, target.dtype, "smap.shape", smap_loaded.shape, smap_loaded.dtype)

        simulated_kspace_21, recon_fromsim_21, traj_21, dcomp_21 = RadialSimulation(target = target,
                spokespertime=inp_spokes,
                nt=nt, nc=nc,
                spokelength=spokelength,
                smap=smap_loaded,
                grid_size = grid_size,
                im_size= im_size)

        print("Output dcomp.shape", dcomp_21.shape, dcomp_21.dtype)
        recon_fromsim_21 = torch2numpy(recon_fromsim_21.permute(0,2,3,1))

        #plt.figure(figsize=(10, 10))
        #plt.imshow(np.abs(recon_fromsim_21[10]))
        #plt.show()

        np.random.seed(0)
        if nl is None:
            #determine the noise level if it's not provided by 1/50 of the kspace center
            #nl = extract_kspacecenter(simulated_kspace_21, spokelength=640)/50
            nl = extract_kspacecenter(simulated_kspace_21, spokelength=640)/100# Thu Oct 22 12:46:49 EDT 2020
            print('nl', nl)

        noise = nl*torch.tensor(np.random.normal(size =simulated_kspace_21.shape)).to(device, dtype)

        #--------------save---------------
        savematname = '{}/{}_sim21spokes_origsmap_{}_{:.2e}.mat'.format(savedir, caseidx, csidx, nl)
        savemat(savematname, mdict ={
                'simulated_kspace_21': simulated_kspace_21.cpu().numpy(),
        #        'x_cgrecon': x_cgrecon_hm,
                'simulated_target': simulated_target,
                'target_recombine': target_recombine,
                'mask': masksloaded,
                'smap_complex': smap_complex,
        #        'output': output
                'noise': noise.cpu().numpy(),
                'traj': traj_21.cpu().numpy(),
                'dcomp': dcomp_21.cpu().numpy(),
                'nl': nl
            })

        target_gifname = '{}/target_case{}_cs{}_Recon_iGRASP.gif'.format(savedir, caseidx, csidx)
        save_gif(simulated_target, target_gifname)

        target2_gifname = '{}/targetrecombine_case{}_cs{}_Recon_iGRASP.gif'.format(savedir, caseidx, csidx)
        save_gif(target_recombine, target2_gifname)
    else:
        print('Only reconstruction from simulation file: {}'.format(filetoload))
        loadedsim = loadmat(filetoload)
        simulated_kspace_21 = torch.tensor(loadedsim['simulated_kspace_21'])
        smap_loaded = numpy2torch(loadedsim['smap_complex'])
        smap_loaded = smap_loaded.permute(1,0,2,3).unsqueeze(0)
        smap_nm = normalized_coilsmap(smap_loaded)
        traj_21 = torch.tensor(loadedsim['traj'])
        dcomp_21 = torch.tensor(loadedsim['dcomp'])
        noise = torch.tensor(loadedsim['noise']).to(device, dtype)
        simulated_target = loadedsim['simulated_target']
        target_recombine = loadedsim['target_recombine']
        masksloaded = loadedsim['mask']
        nl = loadedsim['nl'][0][0]
        #print('!!!', nl)
    #-----------recon-----------
    if niter > 0:
        #---------------iGRASP Reconstruction---------------------
        simulated_kspace_21_np = simulated_kspace_21.view(22, 16, 2, 21, 640).cpu().numpy()
        #print(simulated_kspace_21_np.shape)

        simulated_kspace_21_np = np.concatenate([simulated_kspace_21_np[i] for i in range(nt)], axis=2)
        #print(simulated_kspace_21_np.shape)

        #print(traj.shape)

        smap_reest = cal_coil_sensitivities(simulated_kspace_21_np,
                           get_traj(nt*inp_spokes, 640),
                           resolution=320).unsqueeze(0)

        print('smap_reest', smap_reest.shape)

        smap_reest_nm = normalized_coilsmap(smap_reest)
        smap_nm = normalized_coilsmap(smap_loaded)

        if lambda1 is None:
            warnings.warn('Not lambda1 is specified, calculate that as it = 2.5%max(abs(adjnufft(kspaceinoneframe)))')
            Rdmodel = RadialModel(grid_size=grid_size, im_size= im_size).to(device, dtype)
            x_adjn = Rdmodel.adjoint(y = (simulated_kspace_21*torch.sqrt(dcomp_21)).to(device, dtype),
                            k = traj_21.to(device, dtype),
                            coil_sensitivities= smap_nm.to(device, dtype),
                            #w = dcomp_21.to(device, dtype))
                            w = torch.sqrt(dcomp_21).to(device, dtype))#to make a one matrix for dcomp
            print(x_adjn.shape)
            x_adjn_np = torch2numpy(x_adjn, complexdim=1)
            print(x_adjn_np.shape)
            lambda1=2.5e-2*max(np.abs(x_adjn_np.flatten()))
            print('lambda1 =', lambda1)
        elif lambda1 == 0:
            warnings.warn('lambda1 is set to 0, no regularization.')
            lambda1 = None

        lambdalist = [lambda1] #[1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
        tolerancelist = [1e-12] #[1e-6] #[1e-4, 1e-6, 1e-8, 1e-10] #[1e-4, 1e-10]#[1e-4, 1e-5, 1e-6, 1e-8, 1e-9, 1e-10]
        for lbd in lambdalist:
            for tl in tolerancelist:
                gifname = '{}/sim_{}spokes_loadedsmap_case{}_cs{}_Recon__lambda{}_tolerance{}_iterations{}_noise{:.2e}_nodcomp.gif'.format(savedir, inp_spokes, caseidx, csidx, lbd, tl, niter, nl)
                fname = '{}/sim_{}spokes_loadedsmap_case{}_cs{}_Curve__lambda{}_tolerance{}_iterations{}_noise{:.2e}_target_loadedsmap_nodcomp.png'.format(savedir, inp_spokes, caseidx, csidx, lbd, tl, niter, nl)
                fname_nmlz = '{}/sim_{}spokes_loadedsmap_case{}_cs{}_NmlzCurve__lambda{}_tolerance{}_iterations{}_noise{:.2e}_target_loadedsmap_nodcomp.png'.format(savedir, inp_spokes, caseidx, csidx, lbd, tl, niter, nl)
                fname_recombine = '{}/sim_{}spokes_loadedsmap_case{}_cs{}_Curve__lambda{}_tolerance{}_iterations{}_noise{:.2e}_targetrecombine_loadedsmap_nodcomp.png'.format(savedir, inp_spokes, caseidx, csidx,lbd, tl, niter, nl)
                fname_recombine_nmlz = '{}/sim_{}spokes_loadedsmap_case{}_cs{}_NmlzCurve__lambda{}_tolerance{}_iterations{}_noise{:.2e}_targetrecombine_loadedsmap_nodcomp.png'.format(savedir, inp_spokes, caseidx, csidx,lbd, tl, niter, nl)
                savename = '{}/sim_{}spokes_loadedsmap_case{}_cs{}_Result__lambda{}_tolerance{}_iterations{}_noise{:.2e}_loadedsmap_nodcomp.mat'.format(savedir, inp_spokes, caseidx, csidx, lbd, tl, niter, nl)
                print(fname);
                dcomp_trivial = torch.ones(dcomp_21.shape).to(device, dtype)
                if False:
                    x_cgrecon_NCG = RadialRecon_alternative(kspace=simulated_kspace_21.to(device, dtype)+noise,
                            traj = traj_21.to(device, dtype),
                            #coil_sensitivities = smap.to(device, dtype),
                            coil_sensitivities = smap_nm.to(device, dtype),
                            #coil_sensitivities = smap_reest_nm.to(device, dtype),
                            #w = dcomp_21.to(device, dtype),
                            w = dcomp_trivial,
                            grid_size = grid_size,
                            im_size = im_size,
                            tolerance = tl,
                            lambda1 = lbd,
                            device = device,
                            dtype = dtype,
                            keep_history = True,
                            niter = niter,
                            optimizer = opt)

                x_cgrecon_NCG= RadialRecon(kspace=((simulated_kspace_21).to(device, dtype)+noise),
                            traj = traj_21.to(device, dtype),
                            #coil_sensitivities = smap.to(device, dtype),
                            coil_sensitivities = smap_nm.to(device, dtype),
                            #coil_sensitivities = smap_reest_nm.to(device, dtype),
                            #w = dcomp_trivial.to(device, dtype),
                            w = dcomp_21.to(device, dtype),
                            grid_size = grid_size,
                            im_size = im_size,
                            tolerance = tl,
                            lambda1 = lambda1,
                            device = device,
                            dtype = dtype,
                            keep_history = True,
                            niter = 32)
                if ifhistory:
                    savemat(savename,
                        {'recon': x_cgrecon_NCG,
                        'initial_recon': x_cgrecon_NCG[1],
                        'history': np.array(x_cgrecon_NCG[2]),
                        'simulated_target': simulated_target,
                        'target_recombine': target_recombine,
                        'maskdict': masksloaded,
                        'smap_nm': smap_nm.cpu().numpy(),
                        #'smap': smap.cpu().numpy(),
                        'smap': smap_loaded.cpu().numpy(),
                        'smap_reest': smap_reest.cpu().numpy(),
                        'smap_reest_nm': smap_reest_nm.cpu().numpy(),
                        'noise': noise.cpu().numpy(),
                        'noiselevel': nl})
                else:
                    savemat(savename,
                        {'recon': x_cgrecon_NCG,
                        'initial_recon': x_cgrecon_NCG[1],
                        #'history': np.array(x_cgrecon_NCG[2]),
                        'simulated_target': simulated_target,
                        'target_recombine': target_recombine,
                        'maskdict': masksloaded,
                        'smap_nm': smap_nm.cpu().numpy(),
                        #'smap': smap.cpu().numpy(),
                        'smap': smap_loaded.cpu().numpy(),
                        'smap_reest': smap_reest.cpu().numpy(),
                        'smap_reest_nm': smap_reest_nm.cpu().numpy(),
                        'noise': noise.cpu().numpy(),
                        'noiselevel': nl})

                #save_gif(x_cgrecon_NCG[0],
                #        fname = gifname,
                #        clippoint = 0.8,

                save_gif(x_cgrecon_NCG[0][:,0]+1j*x_cgrecon_NCG[0][:,1], gifname)

                if False:
                    aif_mask = np.flipud(masksloaded['muscle_blood'][0][0] +    masksloaded['glandular_blood'][0][0])
                    heart_mask = np.flipud(masksloaded['heart_blood'][0][0])
                    malig_mask = np.flipud(masksloaded['malignant'][0][0])
                    gland_mask = np.flipud(masksloaded['glandular'][0][0])

                    maskdict={'aif': aif_mask,'heart': heart_mask,'malignant lesion': malig_mask,'gland': gland_mask}
        #
                    plotcurves(np.array(x_cgrecon_NCG[0]),
                             simulated_target, maskdict=maskdict,
                             names = ['Recon', 'target'],
                             normalize = False,
                             nbase = 5,
                             savefname=fname)

                    plotcurves(np.array(x_cgrecon_NCG[0]),
                             target_recombine, maskdict=maskdict,
                             names = ['Recon', 'target recombine'],
                             normalize = False,
                             nbase = 5,
                             savefname=fname_recombine)

                    plotcurves(np.array(x_cgrecon_NCG[0]),
                             simulated_target, maskdict=maskdict,
                             names = ['Recon', 'target'],
                             normalize = True,
                             nbase = 5,
                             savefname=fname_nmlz)

                    plotcurves(np.array(x_cgrecon_NCG[0]),
                             target_recombine, maskdict=maskdict,
                             names = ['Recon', 'target recombine'],
                             normalize = True,
                             nbase = 5,
                             savefname=fname_recombine_nmlz)

if __name__ == "__main__":
    main()

