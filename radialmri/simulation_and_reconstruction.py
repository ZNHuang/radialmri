import sys
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch

import imageio
from skimage import data
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat, savemat

from torchkbnufft import KbNufft as TorchKbNufft
from torchkbnufft import AdjKbNufft as TorchAdjKbNufft
from torchkbnufft import MriSenseNufft, AdjMriSenseNufft

import complex_operations as cpo
#from utils import cg

dtype = torch.float
#device = torch.device('cuda')
device = torch.device('cpu')
warnings.warn('Out of sync with the remove repository')

def numpy2torch(x, device=torch.device('cpu')):
    """
    Cast numpy array to torch tensor.

    Keyword arguments:

    x -- input numpy array
    device -- torch.device, torch.device('cpu') or torch.device('cuda')
        If the input array is not numpy.complex128, numpy.complex64, or
        np.complex_. It's real, complex and imaginary part is going to
        be the same in output.
    """
    if (x.dtype != np.complex128 and x.dtype != np.complex64
            and x.dtype != np.complex_):#TODO
        return torch.tensor(np.array([x, x]), device=device)

    x = torch.tensor(np.array([x.real, x.imag]), device=device)
    return x

def torch2numpy(tensor, complexdim = -1):
    """Casting torch tensor to numpy array

    Keyword arguments:

    tensor -- torch.tensor
    complexdim -- int, the index of complex dimension, either -1 or 1

    """
    nparray = tensor.cpu().detach().numpy()
    assert nparray.shape[complexdim] == 2

    if complexdim == -1:
        return nparray[...,0] + 1j * nparray[...,1]
    elif complexdim == 1:
        return nparray[:,0,...] + 1j * nparray[:,1,...]

    raise NotImplementedError

def get_traj(nspokes, spokelength, keep_last=None):
    """
    Generate kspace trajectory.

    Keyword arguments:

    nspokes -- number of spokes(int)
    spokelength -- length of spokes/readout(int)
    keeplast --

    """#TODO
    ga = 180/((1 + np.sqrt(5))/2)
    ga = np.deg2rad(ga)
    kx = np.zeros(shape=(spokelength, nspokes))
    ky = np.zeros(shape=(spokelength, nspokes))
    ky[:, 0] = np.linspace(-np.pi, np.pi, spokelength)
    for i in range(1, nspokes):
#apply the rotation matrix https://en.wikipedia.org/wiki/Rotation_matrix
        kx[:, i] = np.cos(ga) * kx[:, i - 1] - np.sin(ga) * ky[:, i - 1]
        ky[:, i] = np.sin(ga) * kx[:, i - 1] + np.cos(ga) * ky[:, i - 1]

    ky = np.transpose(ky)
    kx = np.transpose(kx)

    if keep_last: # TODO remove if unnecessary
        kx = kx[-keep_last:]
        ky = ky[-keep_last:]

    traj = np.stack((ky.flatten(), kx.flatten()), axis=0)
    traj = torch.tensor(traj, dtype=dtype, device=device).unsqueeze(0)
    return traj

def density_compensation(total_spokes, spokes_per_frame, readout_length,
        device, minimum =None):
    """
    Calculate the density compensation matrix.

    Keyword Arguments:

    total_spokes -- total number spokes use to reconstruct
    spokes_per_frame -- number of spokes to be used to reconstruct EACH
        FRAME
    readout_length -- size of each readout
    device -- torch.device, torch.device('cpu') or torch.device('cuda')
    minimum -- float, manually define the minimum in the density
        compensation
    """
    bsize = 1

    if minimum == None:
        minimum = 1 / spokes_per_frame

    res = torch.zeros(bsize, readout_length*total_spokes)

    for i in range(int(readout_length/2)):
        res[0, i] = (-(i-readout_length/2)*(1-minimum)*2/readout_length)+minimum

    for i in range(int(readout_length/2), readout_length):
        res[0, i] = ((i-readout_length/2)*(1-minimum)*2/readout_length)+minimum

    for i in range(1, int(total_spokes)):
        res[0,i*readout_length:(i+1)*readout_length] = res[0, :readout_length]

    return res.to(device)

def density_compensation_zerominimum(total_spokes, spokes_per_frame, readout_length,
        device):
    """
    Calculate the density compensation matrix.

    Keyword Arguments:

    total_spokes -- total number spokes use to reconstruct
    spokes_per_frame -- number of spokes to be used to reconstruct EACH
        FRAME
    readout_length -- size of each readout
    device -- torch.device, torch.device('cpu') or torch.device('cuda')
    minimum -- float, manually define the minimum in the density
        compensation
    """
    bsize = 1

    minimum = 0

    res = torch.zeros(bsize, readout_length*total_spokes)

    for i in range(int(readout_length/2)):
        res[0, i] = (-(i-readout_length/2)*(1-minimum)*2/readout_length)+minimum

    for i in range(int(readout_length/2), readout_length):
        res[0, i] = ((i-readout_length/2)*(1-minimum)*2/readout_length)+minimum

    for i in range(1, int(total_spokes)):
        res[0,i*readout_length:(i+1)*readout_length] = res[0, :readout_length]

    return res.to(device)

def cal_coil_sensitivities(kspace, trajt, resolution,\
                           spokes_per_frame = 21, spokelength = 640):
    '''calculate the coil sensitivities by coil_image/rss_image'''
    nch, ri, nspokes_total, spokelength = kspace.shape

    dcomp_numeric = density_compensation(1, spokes_per_frame,
            spokelength, device, minimum=(1/spokes_per_frame)**2)#new DC
    dcomp_numeric = dcomp_numeric.cpu().numpy()[0] #new DC

    kspace = np.expand_dims(kspace,0)#needs a batch dim
    kspace = kspace*dcomp_numeric #apply density compensation
    kspacet = np.reshape(kspace,[1,nch,2,-1])
    kspacet = torch.from_numpy(kspacet)
    kspacet = kspacet.to(dtype=torch.float).to(device)

    layer = TorchAdjKbNufft((resolution,resolution), norm='ortho',
            matadj=True) #adjoint nufft from kspace to image
    layer = layer.to(dtype=torch.float).to(device)
    image_temp = layer(kspacet,trajt).to(torch.device('cpu'))
    #note the precalculated traj is the second input for the adjoint nufft

    real = image_temp[:,:,0,:,:]
    imag = image_temp[:,:,1,:,:]
    coils = (real.numpy() +1j*imag.numpy())
    coils = np.squeeze(coils)
    rss_recon = np.sqrt(np.sum(np.abs(coils)**2,0))

    # sum of squares recon
    rss_recon = np.squeeze(rss_recon)
    coilsmap = np.zeros([nch,2,resolution,resolution])

    for c in range(nch):
        #calculation of the coil sensitivities
        #coilimage/root_sum_of_squaire
        temp= np.squeeze(coils[c,:,:])/(abs(rss_recon))
        #check to make sure these look reasonable
        real_ = gaussian_filter(np.real(temp),5)
        imaginary = gaussian_filter(np.imag(temp),5)
        coilsmap[c,0,:,:] = real_
        coilsmap[c,1,:,:] = imaginary

    coilsmap = torch.tensor(coilsmap, dtype=dtype)
    return coilsmap

def plot_gif(list_img, i, clippoint = 0.8, figsize = (10, 10)):
    """Plot a real image
    Args:
        list_img - A list/array of real image
        i - the index of the image to plot in the list
        clippoint -  the highest pixel value relevant to the maximum
        pixel value, float > 0 and < 1.0
        """
    vmin = min([x.min() for x in list_img])
    vmax = max([x.max() for x in list_img])
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(list_img[i], cmap = 'gray', vmin = vmin,
            vmax = clippoint * vmax)
    ax.set(title='time {}'.format(i))
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image

def save_gif(list_img, fname, clippoint = 0.8, fps = 12,
        figsize = (10, 10), showbutnotsave = False):
    """
    Plot a complex image in terms of is modulus

    Keyword Arguments:

    list_img - A list/array of real image
    fname - file name to save
    clippoint -  the highest pixel value relevant to the maximum
        pixel value
    fps - frame per second
    """
    result = [plot_gif(np.abs(list_img), i, clippoint, figsize) for
            i in range(0,len(list_img))]

    if showbutnotsave:
        return
    imageio.mimsave(fname, result, fps=12)

def plotgif_all(image, prefix, clippoint = 0.8, showbutnotsave = False):
    """
    save all images from the validation set to gifs

    Keyword Arguments:
        image - numpy array, size n*nt*x*y
        prefix - string, prefix for the name of saved images
    """
    for j in range(image.shape[0]):
        save_gif([image[j, i] for i in range(image.shape[1])],
                fname = prefix+"_index_{}.gif".format(j),
                clippoint = clippoint,
                showbutnotsave = showbutnotsave)

def curve_normalized(curve, staticframes):
    m = np.mean(curve[:staticframes])
    return curve/m;

def plotcurves(x_hat, x_gt, maskdict, names = ['Recon', 'Ground Truth'],
        normalize = False, nbase = 5, savefname = None,
        complexsum = False):
    """
    Plot the curve of contrast based on the average signal in the mask
    provided

    Keyword arguments:

    x_ha -- the reconsturcted image, numpy array
    x_gt -- the ground truth, numpy array
    maskdict -- the mask dictionary{'string': numpy array}, 4 masks are
        required
    names -- the name for x_hat and x_gt, list of 2 strings
    normalize -- Normalized the curve or not based on the baseline,
        boolean, default False
    nbase -- number of baseline image frames, int, default 5
    savefname -- directory to say the curve, default None for display
        only(no saving)
    complexsum -- summing the signal in the ROI(mask) as complex values,
        boolean default to be False(summing the signal as their modulus)
    """
    if not complexsum:
        grasp_reconst = np.abs(x_hat)
        magn_target = np.abs(x_gt)
    else:
        grasp_reconst = x_hat
        magn_target = x_gt
        warnings.warn(
                'Calculating the complex sum of signal in the mask!!!')

    nt = x_hat.shape[0]

    k=[]
    mask=[]
    for i in maskdict.keys():
        k.append(i)
        mask.append(maskdict[i])

    #print(list_sigma[1])
    #grasp_reconst = np.abs([temporal_coilimg[]])
    #magn_target = np.array([temporal_coilimg[:,i]*np.conj(smap_temp[:,:,i]) for i in range(temporal_coilimg.shape[1])])
    #magn_target = np.abs(np.sum(x_gt, axis=0))

    # generate contrast curves
    aif_target = np.abs(np.array([(magn_target[i]*mask[0]).sum()
        /mask[0].sum() for i in range(nt)]))
    aif_grasp = np.abs(np.array([(grasp_reconst[i]*mask[0]).sum()
        /mask[0].sum() for i in range(nt)]))
    healthy_target = np.abs(np.array([(magn_target[i]*mask[1]).sum()
        /mask[1].sum() for i in range(nt)]))
    healthy_grasp = np.abs(np.array([(grasp_reconst[i]*mask[1]).sum()
        /mask[1].sum() for i in range(nt)]))
    lesion_target = np.abs(np.array([(magn_target[i]*mask[2]).sum()
        /mask[2].sum() for i in range(nt)]))
    lesion_grasp = np.abs(np.array([(grasp_reconst[i]*mask[2]).sum()
        /mask[2].sum() for i in range(nt)]))
    heart_target = np.abs(np.array([(magn_target[i]*mask[3]).sum()
        /mask[3].sum() for i in range(nt)]))
    heart_grasp = np.abs(np.array([(grasp_reconst[i]*mask[3]).sum()
        /mask[3].sum() for i in range(nt)]))

    time = [x for x in range(nt)]
    print(aif_target.shape)
    if normalize:
        aif_target = curve_normalized(aif_target, nbase)
        aif_grasp = curve_normalized(aif_grasp, nbase)
        healthy_target = curve_normalized(healthy_target, nbase)
        healthy_grasp = curve_normalized(healthy_grasp, nbase)
        lesion_target = curve_normalized(lesion_target, nbase)
        lesion_grasp = curve_normalized(lesion_grasp, nbase)
        heart_target = curve_normalized(heart_target, nbase)
        heart_grasp = curve_normalized(heart_grasp, nbase)

    f, axarr = plt.subplots(1, 4, figsize=(20,10))
    plt.gray()
    axarr[0].plot(time, aif_target, label=names[1])
    axarr[0].plot(time, aif_grasp, label=names[0])
    axarr[0].legend(loc='lower right')
    axarr[0].set_xlabel('Scan time (s)')
    axarr[0].set_ylabel('Signal intensity (a. u.)')
    axarr[0].set_title(k[0], size=14)

    axarr[1].plot(time, healthy_target, label=names[1])
    axarr[1].plot(time, healthy_grasp, label=names[0])
    axarr[1].legend(loc='lower right')
    axarr[1].set_xlabel('Scan time (s)')
    axarr[1].set_ylabel('Signal intensity (a. u.)')
    axarr[1].set_title(k[1], size=14)

    axarr[2].plot(time, lesion_target, label=names[1])
    axarr[2].plot(time, lesion_grasp, label=names[0])
    axarr[2].legend(loc='lower right')
    axarr[2].set_xlabel('Scan time (s)')
    axarr[2].set_ylabel('Signal intensity (a. u.)')
    axarr[2].set_title(k[2], size=14)

    axarr[3].plot(time, heart_target, label=names[1])
    axarr[3].plot(time, heart_grasp, label=names[0])
    axarr[3].legend(loc='lower right')
    axarr[3].set_xlabel('Scan time (s)')
    axarr[3].set_ylabel('Signal intensity (a. u.)')
    axarr[3].set_title(k[3], size=14)

    if savefname is not None:
        f.savefig(savefname)

    plt.show()

def normalized_coilsmap(smap):
    """Normalized the coil sensitivities"""
    print(smap.shape)
    #calculated the modulus
    temp = cpo.multiplication_conjugate(smap, smap, dim=2)
    modulus = torch.sum(temp, dim=1)
    #copy the real part to the imaginary,
    #for convenience during broadcasting
    modulus[:,1] = modulus[:,0]

    smap_nm = smap/modulus
    #print(smap_nm.shape)
    return smap_nm

#TODO remove
def plotcoil(x, n=8, cmap = 'gray'):
    """
    Plot a 3D numpy array as images in subplot in their modulus
    Keyword arguments:
        x -- numpy array, can be complex, dimension: [slice, n, m]
        n -- int, number of slices to show, default 8
        cmap -- matplotlib cmap parameter
    """
    f,axes =plt.subplots(1, n, figsize = (32, 4))
    for i in range(n):
        axes[i].imshow(np.abs(x[i]), cmap = cmap)

    plt.show()

#TODO remove
def plotloss(output, iterations, savefname = None):
    f = plt.figure()
    plt.plot(list(range(iterations)), [np.log10(float(i.split(' ')[-1]))
        for i in output[1:iterations*2:2]])
    if savefname is not None:
        f.savefig(savefname);
    else:
        plt.show()

def sim_coil(imageframe, ss, coild = 2 ):
    '''Args:
        imageframe: n*m (complex) numpy array
        ss: coil sensitivities n*m*coilcount or coilcount*n*m coil
        sensitivities numpy array (complex)
        coild: coil dimension in coil sensitivities int 2 or 0'''
    if coild ==2:
        coil_image = np.array([imageframe*ss[:,:,i] for i in range(ss.shape[coild])])
    elif coild == 0:
        coil_image = np.array([imageframe*ss[i,:,:] for i in range(ss.shape[coild])])
    else:
        raise NotImplementedError

    return coil_image

#TODO remove
def coil_combine(coilimages, ss, coild = 2):
    """
    Combine coil images to a single output

    Keyword Arguments:
    coilimages: n*m*coilcount or coilcount*n*m (complex) numpy array
    ss: coil sensitivities n*m*coilcount or coilcount*n*m coil
    sensitivities (complex) numpy array
    coild: coil dimension in coil sensitivities int 2 or 0
    """
    if coild == 2:
        return np.sum(np.array([coilimages[:,:,i]*np.conj(ss[:,:,i]) \
                      for i in range(ss.shape[coild])]), axis=coild)
    elif coild == 0:
        return np.sum(np.array([coilimages[i,:,:]*np.conj(ss[i,:,:]) \
                      for i in range(ss.shape[coild])]), axis = coild)
    else:
        raise NotImplementedError


def RadialSimulation(target, spokespertime, nt, nc, spokelength, \
                     smap, grid_size, im_size, dcompmatrix = None):
    """
    Radial kspace simulation
    Keywoard Arguments:
        target -- pytorch tensor of shape [nt, 2, n, m]
        spokespertime -- int
        nt -- number of frames
        nc -- number of coils
        spokelength -- readout points count on each spoke
        smap -- coil sensitvities maps, pytorch tensor of shape
            [1, nc, 2, n, m]

        grid_size -- gridding grid size used in NUFFT, tuple of length 2
        im_size -- image size, tuple equals to (n, m)

        dcompmatrix -- torch tensor of size [nt, nc, 2, spokelength * nt]
    """

    #---Basics---
    spokes_per_frame_fs = spokespertime
    nspokes_sim_fs = spokes_per_frame_fs*nt
    #recon_np_fs = torch2numpy(recon_fromsim_fs.permute(0,2,3,1))

    temp = get_traj(nspokes_sim_fs, spokelength)

    #---sampling Trajectory---
    traj_sim_fs = torch.zeros(nt, 2, spokes_per_frame_fs*spokelength)
    for i in range(nt):
        traj_sim_fs[i] = temp[0,:,spokes_per_frame_fs*i*spokelength:\
                              spokes_per_frame_fs*(i+1)*spokelength]

    #print(traj_sim_fs.shape, traj_sim_fs.shape)

    #---Density Compensation---
    if dcompmatrix is None:
        dcomp_fs = density_compensation(spokes_per_frame_fs, \
                                        spokes_per_frame_fs, \
                                        spokelength, device, \
                                        minimum=(1/spokes_per_frame_fs)**2)
        dcomp_fs = dcomp_fs.unsqueeze(0).unsqueeze(0)
        dcomp_fs = dcomp_fs.repeat(nt, nc, 2, 1)
    else:
        dcomp_fs = dcompmatrix

    print(dcomp_fs.shape)

    #---Calculation---
    #torch.cuda.empty_cache()
    RdModel = RadialModel(grid_size, im_size).to(device, dtype)

    simulated_kspace_fs =RdModel.forward(x = target.to(device, dtype),
                    k = traj_sim_fs.to(device, dtype),
                    coil_sensitivities=smap.to(device, dtype),
                    w= torch.ones(dcomp_fs.shape).to(device, dtype))#remove the multiplication of the sqrt(dcomp) in the simulation by replacing the dcomp_fs with matrix of ones. Mon Oct 19 10:51:09 EDT 2020
                    #w= dcomp_fs.to(device, dtype))

    print(simulated_kspace_fs.shape)

    #recon_fromsim_fs = RdModel.adjoint(y = simulated_kspace_fs,
    recon_fromsim_fs = RdModel.adjoint(y = simulated_kspace_fs*torch.sqrt(dcomp_fs.to(device, dtype)),#Mon Oct 19 10:56:53 EDT 2020
                    k = traj_sim_fs.to(device, dtype),
                    coil_sensitivities=smap.to(device, dtype),
                    w= dcomp_fs.to(device, dtype))

    print(recon_fromsim_fs.shape)

    return simulated_kspace_fs, recon_fromsim_fs, traj_sim_fs, dcomp_fs

class CartesianModel(torch.nn.Module):
    """
    Cartesian acquistion and reconstruction

    Keyword Arguments:

    x -- the image
    coil_sensitivities -- need to have the same dimension as the
    reconstructed/original image
    """
    def __init__(self):
        super()
        #super(CartesianModel, self).__init__()

    def forward(self, x, coil_sensitivities):
        """FFT after applying coil sensitivities
        Args:
            x: torch tensor of dimension(nt, 2, x, y)
            coil_sensitivities: torch tensor of dimension (ncoil, nt, 2, x, y)
        Returns:
            y: (ncoil, nt, x, y, 2)
        """

        cimage = torch.zeros(coil_sensitivities.shape, dtype=coil_sensitivities.dtype)
        for i in range(coil_sensitivities.shape[0]):
            cimage[i] = cpo.multiplication(x.squeeze(), coil_sensitivities[i], dim=1)
            #Mon Dec 13 17:48:17 EST 2021
        #cimage = cpo.multiplication(x.unsqueeze(1), coil_sensitivities, dim =2)

        #cimage = cimage.permute(0, 1, 3, 4, 2) #from (ncoil, nt, 2, x, y) to (ncoil, nt, x, y, 2)
        cimage = cimage.permute(1, 0, 3, 4, 2) #from (ncoil, nt, 2, x, y) to (nt, ncoil, x, y, 2)
        y = torch.fft(cimage, signal_ndim=2, normalized=True)
        return y

    def backward(self, y, coil_sensitivities):
        """iFFT and coil sensitivities combine
        Args:
            y, multicoil kspace data, tensor of dimension (ncoil, nt, kx, ky, 2)
            coil_sensitivities: torch tensor of dimension (ncoil, nt, 2, x, y)

        Returns:
            x, coil sensitivities combined image
        """
        #print('y.shape', y.shape)
        cimage = torch.ifft(y, signal_ndim=2, normalized=True)
        #print('cimage, coil_sensitivities.shape', cimage.shape, cimage.dtype, cimage.device, coil_sensitivities.shape, coil_sensitivities.dtype, coil_sensitivities.device)
        cimage = cimage.permute(1, 0, 4, 2, 3)
        image = torch.sum(cpo.multiplication_conjugate(cimage.to(coil_sensitivities.device), coil_sensitivities, dim=2), dim=1, keepdim=True);
        return image.squeeze()

    def tv_loss(self):
        raise NotImplementedError;


class CartesianModel2(torch.nn.Module):
    """Cartesian acquistion and reconstruction
        x: the image
        coil_sensitivities: need to have the same dimension as the reconstructed/original image
    """

    def __init__(self, coil_sensitivities, mask):
        super(CartesianModel2, self).__init__()
        self.coil_sensitivities = coil_sensitivities
        self.mask = mask

    def forward(self, x):
        cimage = torch.zeros(self.coil_sensitivities.shape, dtype=self.coil_sensitivities.dtype)
        for i in range(self.coil_sensitivities.shape[0]):
            cimage[i] = cpo.multiplication(x.squeeze(), self.coil_sensitivities[i], dim=2)
        y = torch.fft(cimage, signal_ndim=2, normalized=True)
        return y

    def backward(self, y):
        cimage = torch.ifft(y, signal_ndim=2, normalized=True)
        image = torch.sum(cpo.multiplication_conjugate(cimage, self.coil_sensitivities, dim=3), dim=0, keepdim=True)
        return image

    def weight(self, y):
        return y*self.mask

def CartesianSimulation(target, smap):
    """Cartesiam kspace simulation"""
    #torch.cuda.empty_cache()
    RdModel = CartesianModel().to(device, dtype)

    simulated_kspace_fs = RdModel.forward(x = target.to(device, dtype),
                    coil_sensitivities=smap.to(device, dtype))

    print(simulated_kspace_fs.shape)

    recon_fromsim_fs = RdModel.backward(y = simulated_kspace_fs,
                    coil_sensitivities=smap.to(device, dtype))

    print(recon_fromsim_fs.shape)

    return simulated_kspace_fs, recon_fromsim_fs

def CartesianRecon(kspace, coil_sensitivities, w, tolerance = 0.001,
                   lambda1 = None, lambda2 = None, device=device,
                   dtype=dtype, keep_history = False, niter=12,
                   optimizer = 'GD', stepsize = None, verbose = True):
    #TODO

    """
    Cartesian kspace reconstruction.
    tolerance is used to determine stop condition. It is to be tuned,
    original setting was 0.01, however 0.001 seems to give a better
    curve(R=2) Implementing CG according to
    https://dl.acm.org/doi/pdf/10.1145/3180496.3180632
    w is the sampling mask

    Returns:
        reconstruction, zero-filled recon, history(if keep_history = True)
    """

    model = CartesianModel()
    model = model.to(device, dtype)

    initial_recon = model.backward(kspace, coil_sensitivities)

    #------------Radial specific------------

    #print('initial_recon.shape', initial_recon.shape)
    output = model.forward(initial_recon, coil_sensitivities)
    target = kspace
    scalar = (output*target).sum()/(output**2).sum()
    x0 = (scalar.detach()*initial_recon).requires_grad_(True)
    initial_recon = x0.cpu().detach().numpy().copy()

    history = []

    #------------optimization----------------
    if stepsize is None:
        stepsize = [0.5]*15+[0.05]*niter
    if optimizer == 'GD':
        x0 = model.backward(kspace, coil_sensitivities)
        #print('x0.shape', x0.shape)
        #print('w.shape', w.shape)
        with torch.no_grad():
            for i in range(niter):
                output = model.forward(x0, coil_sensitivities) * w

                #print('output.shape', output.shape)
                residual = (output - kspace)
                #print(i, 'Residual l2 norm ={:f}'.format(torch.norm(residual)))
                x0 = x0 - stepsize[i] * model.backward(residual, coil_sensitivities)
                history.append(x0.detach().cpu().numpy())
    elif optimizer == 'CG5':
        EH_b = model.backward(kspace, coil_sensitivities)
        x0 = torch.zeros(EH_b.shape).to(device)
        r = EH_b.detach().clone()
        p = EH_b.detach().clone()
        with torch.no_grad():
            for i in range(niter):
                rHr = torch.norm(r)**2
                E_p = model.forward(p, coil_sensitivities) * w
                q = model.backward(E_p, coil_sensitivities) # E^H E p

                pHq = torch.sum(cpo.multiplication_conjugate(q, p, dim=1), dim =(0, 2, 3))
                pHqconj = pHq.detach().clone()
                pHqconj[1] = -pHqconj[1]
                #print(pHq, pHqconj)
                oneoverpHq = pHqconj/(torch.norm(pHq)**2)
                #print('phq.shape', pHq.shape)
                #print('oneoverpHp.shape', oneoverpHp.shape)
                alpha = rHr * oneoverpHq
                if verbose:
                    print(i, 'alpha: ', alpha)
                alpha_repeat = torch.ones(x0.shape).to(device)
                alpha_repeat[:, 0] = alpha_repeat[:, 0]*alpha[0]
                alpha_repeat[:, 1] = alpha_repeat[:, 1]*alpha[1]
                x0 = x0 + cpo.multiplication(alpha_repeat, p, dim=1)
                r_new = r - cpo.multiplication(alpha_repeat, q, dim=1)
                beta = (torch.norm(r_new)/torch.norm(r))**2
                p = r_new + beta*p
                r = r_new
                history.append(x0.detach().cpu().numpy())
    else:
        print('optimizer is not defined:', optimizer)
        print('Please only use CG5 or GD as input for optimizer')
        raise NotImplementedError


#    for i in range(niter):
#        print('iteration ', i)
#        loss = optimizer.step(criterion);
#        print('loss= {}'.format(float(loss)))
#        if keep_history:
#            history.append(x0.cpu().detach().numpy().copy())
#
#        if abs(last_loss - loss) < stop_criterion:
#            print('\tProgress is smaller than tolerance, stop optimization')
#            break
#        last_loss = loss

    if keep_history:
        return x0.cpu().detach(), initial_recon, history
    else:
        return x0.cpu().detach(), initial_recon

class RadialModel(torch.nn.Module):
    def __init__(self, grid_size, im_size):#, device=device, dtype=dtype):
        super().__init__()
        self.nufft_op = TorchKbNufft(grid_size=grid_size, \
                                     im_size=im_size, \
                                     numpoints=(6, )*len(grid_size), \
                                     norm = 'ortho', \
                                     matadj=True)#.to(device, dtype)

        self.adjnufft_op = TorchAdjKbNufft(grid_size=grid_size, \
                                           im_size=im_size, \
                                           numpoints=(6, )*len(grid_size), \
                                           norm = 'ortho', \
                                           matadj=True)#.to(device, dtype)

    def forward(self, x, k, coil_sensitivities, w):
        """Args:
            x: image(combined in coil dimension), (nt, 2, nx, ny)
            k: kspace traj (nt, 2, (spokecount* spokelength))
            coil_sensitivities: (1, nc, 2, nx, ny)
            w: desity compensation, (nt, nc, 2, (spokecount*spokelength))
           Returns:
            y: Radial kspace, (nt, nc, 2, (spokecount*spokelength))
            """
        #assert coil_sensitivities.shape[0] == 1
        #assert coil_sensitivities.shape[2] == 2
        cimage = cpo.multiplication(x.unsqueeze(1), coil_sensitivities, dim =2)
        y = self.nufft_op(cimage, k)
        y = y* torch.sqrt(w)
        return y

    def adjoint(self, y, k, coil_sensitivities, w):
        """Args:
            y: Radial kspace, (nt, nc, 2, (spokecount*spokelength))
            k: kspace traj (nt, 2, (spokecount* spokelength))
            coil_sensitivities: (1, nc, 2, nx, ny)
            w: desity compensation, (nt, nc, 2, (spokecount*spokelength))
          Return:
            x: image(combined in coil dimension), (nt, 2, nx, ny)
        """
        y = y* torch.sqrt(w)
        cimage = self.adjnufft_op(y, k)
        x = cpo.multiplication_conjugate(cimage, coil_sensitivities, dim = 2)
        x = x.sum(1)
        return x

    def adjoint_coilimage(self, y, k, w):
        """Return the coil images withou sensitivities combine
            Args:
            y: Radial kspace, (nt, nc, 2, (spokecount*spokelength))
            k: kspace traj (nt, 2, (spokecount* spokelength))
            w: desity compensation, (nt, nc, 2, (spokecount*spokelength))
          Return:
            x: image(combined in coil dimension), (nt, nc, 2, nx, ny)
        """
        y = y* torch.sqrt(w)
        cimage = self.adjnufft_op(y, k)
        return cimage

    def TV_loss(self, x, l1Smooth =1E-15, lambda1=None, lambda2 = None):
        """Why is this l1Smooth needed?"""
        TV_loss = torch.zeros([], requires_grad=True).to(x.device, x.dtype)
        if lambda1 is not None:
            #temporal regularization
            delta = x[1:] - x[:-1]
            tloss = delta[:,0]**2+delta[:,1]**2+l1Smooth
            TV_loss = TV_loss+lambda1*torch.sqrt(tloss).sum()

        if lambda2 is not None:#TODO not tested yet
            #spacial regularization
            delta_2 = x[:,:,1:,:-1] - x[:,:,:-1,:-1]
            delta_3 = x[:,:,:-1,1:] - x[:,:,:-1,:-1]
            delta = torch.stack((delta_2, delta_3), dim=2)
            sloss = delta[:,0]**2 + delta[:,1]**2+l1Smooth
            TV_loss = TV_loss+lambda2*torch.sqrt(sloss).sum()
        #print('TV_loss=', TV_loss)
        return TV_loss;

class RadialModel_IC(torch.nn.Module):
    def __init__(self, grid_size, im_size, IC):#, device=device, dtype=dtype):
        super().__init__()
        self.nufft_op = TorchKbNufft(grid_size=grid_size, \
                                     im_size=im_size, \
                                     numpoints=(6, )*len(grid_size), \
                                     norm = 'ortho', \
                                     matadj=True)#.to(device, dtype)

        self.adjnufft_op = TorchAdjKbNufft(grid_size=grid_size, \
                                           im_size=im_size, \
                                           numpoints=(6, )*len(grid_size), \
                                           norm = 'ortho', \
                                           matadj=True)#.to(device, dtype)
        self.IC = IC #if to do intensity correction

    def forward(self, x, k, coil_sensitivities, I, w):
        """
        Fowrad operator

        Keyword Arguments:
            x: image(combined in coil dimension), (nt, 2, nx, ny)
            k: kspace traj (nt, 2, (spokecount* spokelength))
            coil_sensitivities: (1, nc, 2, nx, ny)
            I: (nt, 2, nx, ny) for intensity correction
                (introduced by coil sensitivities),
                1 /(sqrt(coil_sensitivities**2 sum over coil)) repeated
                in the nt dimension. Only real dimension are nonzero
            w: desity compensation, (nt, nc, 2, (spokecount*spokelength))

        Returns:
            y: Radial kspace, (nt, nc, 2, (spokecount*spokelength))
        """
        #assert coil_sensitivities.shape[0] == 1
        #assert coil_sensitivities.shape[2] == 2
        if self.IC:
            x = cpo.multiplication(x, I, dim =1)
        cimage = cpo.multiplication(x.unsqueeze(1), coil_sensitivities, dim =2)
        y = self.nufft_op(cimage, k)
        y = y* torch.sqrt(w)
        return y

    def adjoint(self, y, k, coil_sensitivities, I, w):
        """
        ajoint operator

        Keyword Arguments:
            y: Radial kspace, (nt, nc, 2, (spokecount*spokelength))
            k: kspace traj (nt, 2, (spokecount* spokelength))
            coil_sensitivities: (1, nc, 2, nx, ny)
            I: (nt, 2, nx, ny) for intensity correction
                (introduced by coil sensitivities),
                1 /(sqrt(coil_sensitivities**2 sum over coil)) repeated
                in the nt dimension. Only real dimension are nonzero
            w: desity compensation, (nt, nc, 2, (spokecount*spokelength))

        Return:
            x: image(combined in coil dimension), (nt, 2, nx, ny)
        """
        #y = y* torch.sqrt(w.view(w.shape[0], 1, 1, -1))
        y = y* torch.sqrt(w)
        cimage = self.adjnufft_op(y, k)
        x = cpo.multiplication_conjugate(cimage, coil_sensitivities, dim = 2)
        x = x.sum(1)
        if self.IC:
            x = cpo.multiplication(x, I, dim=1)
        return x

    def adjoint_coilimage(self, y, k, w):
        """
        Return the coil images withou sensitivities combine

        Args:
            y: Radial kspace, (nt, nc, 2, (spokecount*spokelength))
            k: kspace traj (nt, 2, (spokecount* spokelength))
            w: desity compensation, (nt, nc, 2, (spokecount*spokelength))

        Return:
            x: image(combined in coil dimension), (nt, nc, 2, nx, ny)
        """
        y = y* torch.sqrt(w)
        cimage = self.adjnufft_op(y, k)
        return cimage

    def TV_loss(self, x, l1Smooth =1E-15, lambda1=None, lambda2 = None):
        """Why is this l1Smooth needed?"""
        TV_loss = torch.zeros([], requires_grad=True).to(x.device, x.dtype)
        if lambda1 is not None:
            #temporal regularization
            delta = x[1:] - x[:-1]
            tloss = delta[:,0]**2+delta[:,1]**2+l1Smooth
            TV_loss = TV_loss+lambda1*torch.sqrt(tloss).sum()

        if lambda2 is not None:#TODO not tested yet
            #spacial regularization
            delta_2 = x[:,:,1:,:-1] - x[:,:,:-1,:-1]
            delta_3 = x[:,:,:-1,1:] - x[:,:,:-1,:-1]
            delta = torch.stack((delta_2, delta_3), dim=2)
            sloss = delta[:,0]**2 + delta[:,1]**2+l1Smooth
            TV_loss = TV_loss+lambda2*torch.sqrt(sloss).sum()
        #print('TV_loss=', TV_loss)
        return TV_loss;

def RadialRecon_alternative(kspace, traj, coil_sensitivities, w,
                            grid_size, im_size,  tolerance = 0.001,
                            lambda1 = None, lambda2 = None, device=device,
                            dtype=dtype, keep_history = False, niter=12,
                            optimizer = 'GD', stepsize = None,
                            verbose = True):
    """
    Radial kspace reconstruction.
    tolerance is used to determine stop condition. It is to be tuned,
    original setting was 0.01, however 0.001 seems to give a better
    curve(R=2) Implementing CG according to
    https://dl.acm.org/doi/pdf/10.1145/3180496.3180632

    Returns:
        reconstruction, NUFFT recon, history(if keep_history = True)

    """

    model = RadialModel(grid_size, im_size)
    model = model.to(device, dtype)

    initial_recon = model.adjoint(kspace*torch.sqrt(w), traj, coil_sensitivities, w)
    #------------Radial specific------------
    output = model.forward(initial_recon, traj, coil_sensitivities, w)
    target = kspace
    scalar = (output*target).sum()/(output**2).sum()
    x0 = (scalar.detach()*initial_recon).requires_grad_(True)
    initial_recon = x0.cpu().detach().numpy().copy()
    #------------Radial specific------------
    #x0 = (initial_recon).requires_grad_(True);

#    def criterion(requires_grad =False):
#        optimizer.zero_grad();
#        output = model.forward(x0, traj, coil_sensitivities, w)
#        #print(initial_recon.device, x0.device, output.device, kspace.device, mask.device)
#        loss = (((output-kspace)) **2).sum()
#
#
#        if lambda1 is not None or lambda2 is not None:
#            loss += model.TV_loss(x0, l1Smooth =1E-15, lambda1=lambda1, lambda2 = lambda2)
#
#        if requires_grad:
#            loss.backward()
#
#        return loss
#
#    config = {'nite': 10, 't0': 1E4, 'maxlsiter': 15}# same
#    optimizer = cg.CG([x0], **config)# same optimizer for iGRASP

#    loss = criterion().to(device)

#    l0=loss
#    stop_criterion = tolerance* l0

    #last_loss = l0
    history = []

    #optimization
    if stepsize is None:
    	stepsize = [0.5]*15+[0.05]*niter
    if optimizer == 'GD':
        x0 = model.adjoint(kspace*torch.sqrt(w), traj, coil_sensitivities, w)
        with torch.no_grad():
            for i in range(niter):
                output = model.forward(x0, traj, coil_sensitivities, w)
                residual = (output - kspace)
                if verbose:
                    print(i, 'Residual l2 norm ={:f}'.format(torch.norm(residual)))
                #x0 = x0 - stepsize[i]*model.adjoint(residual*torch.sqrt(w), traj, coil_sensitivities, w)
                x0 = x0 - stepsize[i]*model.adjoint(residual, traj, coil_sensitivities, w)
                history.append(x0.detach().cpu().numpy())
    elif optimizer == 'CG2':
        #Based on Florian Knoll's implementation in Compressed Sensing tutorial
        #x0#image domain
        x0 = model.adjoint(kspace*torch.sqrt(w), traj, coil_sensitivities, w)
        r = x0.detach().clone()#image domain
        p = r.detach().clone()#image domain
        rr = torch.norm(r)**2 #TODO
        print('coil_sensitivities.shape', coil_sensitivities.shape)
        with torch.no_grad():
            for i in range(niter):
                temp = model.forward(p, traj, coil_sensitivities, w)
                Ap = model.adjoint(temp, traj, coil_sensitivities, w)
                #print(p.shape, Ap.shape)
                a = rr/(cpo.multiplication(p, Ap, dim=1).sum())#step size
                print(i, a)
                x0 = x0 + a*p
                rnew = r - a*Ap
                b = (torch.norm(rnew))/rr
                r = rnew
                rr = torch.norm(r)**2
                p = r + b*p
                #real_residual = model.forward(x0, traj, coil_sensitivities, w)
                #print(i, 'Real Residual l2 norm'.format())
                if verbose:
                    print(i, 'Residual l2 norm ={:f}'.format(rr))
    elif optimizer == 'CG5':
        EH_b = model.adjoint(kspace*torch.sqrt(w), traj, coil_sensitivities, w)
        x0 = torch.zeros(EH_b.shape).to(device)
        r = EH_b.detach().clone()
        p = EH_b.detach().clone()
        with torch.no_grad():
            for i in range(niter):
                rHr = torch.norm(r)**2
                E_p = model.forward(p, traj, coil_sensitivities, w)
                q = model.adjoint(E_p, traj, coil_sensitivities, w) # E^H E p

                pHq = torch.sum(cpo.multiplication_conjugate(q, p, dim=1), dim =(0, 2, 3))
                pHqconj = pHq.detach().clone()
                pHqconj[1] = -pHqconj[1]
                #print(pHq, pHqconj)
                oneoverpHq = pHqconj/(torch.norm(pHq)**2)
                #print('phq.shape', pHq.shape)
                #print('oneoverpHp.shape', oneoverpHp.shape)
                alpha = rHr*oneoverpHq
                if verbose:
                    print(i, 'alpha: ', alpha)
                alpha_repeat = torch.ones(x0.shape).to(device)
                alpha_repeat[:, 0] = alpha_repeat[:, 0]*alpha[0]
                alpha_repeat[:, 1] = alpha_repeat[:, 1]*alpha[1]
                x0 = x0 + cpo.multiplication(alpha_repeat, p, dim=1)
                r_new = r - cpo.multiplication(alpha_repeat, q, dim=1)
                beta = (torch.norm(r_new)/torch.norm(r))**2
                p = r_new + beta*p
                r = r_new
                history.append(x0.detach().cpu().numpy())
    elif optimizer == 'iGRASP':
        raise NotImplementedError
        #the algorithm is the same as CG5, but the operators are different.
        #From solving Ex = b to solve E'x = b'
        #where E' = E^H E + \lambda R^H R, b' = E^H * b, R is the regularization.
        b = model.adjoint(kspace*torch.sqrt(w), traj, coil_sensitivities, w)
        x0 = torch.zeros(b.shape).to(device)
        r = b.detach().clone() #-  #A c # TODO Mon Aug 23 19:48:58 EDT 2021
        p = r.detach().clone()
        with torch.no_grad():
            for i in range(niter):
                rHr = torch.norm(r)**2
                E_p = model.forward(p, traj, coil_sensitivities, w)
                q = model.adjoint(E_p, traj, coil_sensitivities, w) # E^H E p

                pHq = torch.sum(cpo.multiplication_conjugate(q, p, dim=1), dim =(0, 2, 3))
                pHqconj = pHq.detach().clone()
                pHqconj[1] = -pHqconj[1]
                #print(pHq, pHqconj)
                oneoverpHq = pHqconj/(torch.norm(pHq)**2)
                #print('phq.shape', pHq.shape)
                #print('oneoverpHp.shape', oneoverpHp.shape)
                alpha = rHr*oneoverpHq
                if verbose:
                    print(i, 'alpha: ', alpha)
                alpha_repeat = torch.ones(x0.shape).to(device)
                alpha_repeat[:, 0] = alpha_repeat[:, 0]*alpha[0]
                alpha_repeat[:, 1] = alpha_repeat[:, 1]*alpha[1]
                x0 = x0 + cpo.multiplication(alpha_repeat, p, dim=1)
                r_new = r - cpo.multiplication(alpha_repeat, q, dim=1)
                beta = (torch.norm(r_new)/torch.norm(r))**2
                p = r_new + beta*p
                r = r_new
                history.append(x0.detach().cpu().numpy())

    else:
        print('optimizer is not defined:', optimizer)
        print('Please only use CG2, CG5 or GD as input for optimizer')
        raise NotImplementedError


#    for i in range(niter):
#        print('iteration ', i)
#        loss = optimizer.step(criterion);
#        print('loss= {}'.format(float(loss)))
#        if keep_history:
#            history.append(x0.cpu().detach().numpy().copy())
#
#        if abs(last_loss - loss) < stop_criterion:
#            print('\tProgress is smaller than tolerance, stop optimization')
#            break
#        last_loss = loss

    if keep_history:
        return x0.cpu().detach().numpy().copy(), initial_recon, history
    else:
        return x0.cpu().detach().numpy().copy(), initial_recon

def RadialRecon(kspace, traj, coil_sensitivities, w, grid_size, im_size,
                tolerance = 0.001, lambda1 = None, lambda2 = None,
                device=device, dtype=dtype, keep_history = False, niter=12):
    """
    Radial kspace reconstruction.
    tolerance is used to determine stop condition. It is to be tuned, original setting was 0.01, however 0.001 seems to give a better curve(R=2)
    """

    model = RadialModel(grid_size, im_size)
    model = model.to(device, dtype)

    initial_recon = model.adjoint(kspace*torch.sqrt(w), traj,
                                  coil_sensitivities, w)
    #------------Radial specific------------
    output = model.forward(initial_recon, traj, coil_sensitivities, w)
    target = kspace
    scalar = (output*target).sum()/(output**2).sum()
    x0 = (scalar.detach()*initial_recon).requires_grad_(True)
    initial_recon = x0.cpu().detach().numpy().copy()
    #------------Radial specific------------
    #x0 = (initial_recon).requires_grad_(True);

    def criterion(requires_grad =False):
        optimizer.zero_grad();
        output = model.forward(x0, traj, coil_sensitivities, w)
        loss = (((output-kspace)) **2).sum()


        if lambda1 is not None or lambda2 is not None:
            loss += model.TV_loss(x0, l1Smooth =1E-15,
                                  lambda1=lambda1, lambda2 = lambda2)

        if requires_grad:
            loss.backward()

        return loss

    config = {'nite': 10, 't0': 1E4, 'maxlsiter': 15}# same
    optimizer = cg.CG([x0], **config)# same optimizer for iGRASP

    loss = criterion().to(device)

    l0=loss
    stop_criterion = tolerance* l0

    last_loss = l0
    history = []
    for i in range(niter):
        print('iteration ', i)
        loss = optimizer.step(criterion);
        print('loss= {}'.format(float(loss)))
        if keep_history:
            history.append(x0.cpu().detach().numpy().copy())

        if abs(last_loss - loss) < stop_criterion:
            print('\tProgress is smaller than tolerance, stop optimization')
            break
        last_loss = loss

    if keep_history:
        return x0.cpu().detach().numpy().copy(), initial_recon, history
    else:
        return x0.cpu().detach().numpy().copy(), initial_recon

def loadcoilsensitivities(dir, device = device):
    """
    Load and return the simulated coil sensitivities as a torch.tensor

    Keyword Arguments:

        dir -- .mat file directory
        device -- cpu or gpu torch device
    """
    cs_loaded = loadmat(dir)
    smap_loaded = numpy2torch(cs_loaded['imgSens'], device =device)
    smap_loaded = smap_loaded.permute(3, 0, 1, 2).unsqueeze(0)
    return smap_loaded

def extract_kspacecenter(kspace, nt = 22, nc=16, spokelength = 640):
    """
    Extract the center(2 kspace point) of the kspace value, for noise level determination

    Keyword Arguments:

        kspace -- is the shape (nt, ncoil, 2, spokesperframe*spokelength)
        nt -- time points
        nc -- coil counts
        spokelength -- sampling point counts in each spoke
    """
    if spokelength%2 != 0:
        raise NotImplementedError

    kspacesorted = kspace.view(nt, nc, 2, -1, spokelength)
    kspacecat = torch.cat([kspacesorted[i] for i in range(nt)], dim=-2)
    kspacecat.shape

    temp = kspacecat[:,:,:,int(spokelength/2-1):int(spokelength/2+1)]
    temp = torch2numpy(temp.permute(0, 2, 3, 1), complexdim=-1)

    return np.mean(np.abs(temp))

def extract_kspacecenter_cartesian(kspace, nt = 22, nc=16):
    """
    Extract the center(2 kspace point) of the kspace value, for noise level determination

    Keyword Arguments:

        kspace -- is the shape (nt, ncoil, 2, kx, ky)
        nt -- time points
        nc -- coil counts
    """
    _, _, _, nkx, nky = kspace.shape
    if nkx % 2 != 0 or nky % 2 != 0:
        raise NotImplementedError

    temp = kspace[:,:,:, int(nkx / 2 - 1 ) : int(nkx / 2 + 1), \
                  int(nky / 2 - 1) : int(nky / 2 + 1)]
    temp = torch2numpy(temp.permute(0, 1, 3, 4, 2), complexdim=-1)

    #plt.violinplot(dataset=np.abs(temp[0]))
    return np.mean(np.abs(temp))


if __name__ == "__main__":
    pass
