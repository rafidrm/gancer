from __future__ import print_function
import torch
import numpy as np
import torch
from PIL import Image
from skimage.transform import rescale
import os


def vox2tensor(img):
    ''' Converts a Numpy array (D x H x W x C) to a voxel Tensor
    ( C x D x H x W )
    '''
    if img.ndim == 4:
        img = img.transpose((3, 0, 1, 2))
        img = torch.from_numpy(img)
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img
    else:
        raise TypeError('vox should have 4 dimensions.')
    pass


def sparse2iv(mat):
    ''' Converts a Numpy array sparse (W x H) to the necessary pieces
    in order to make a sparse tensor later. Pieces are i, v, and shape.
    '''
    if mat.ndim == 2:
        val = mat.data
        ind = np.vstack((mat.row, mat.col))
        shape = mat.shape

        i = torch.LongTensor(ind)
        v = torch.FloatTensor(val)
        return i, v, shape
    else:
        raise TypeError('mat should have 2 dimensions.')


def sparseivshape2tensor(i, v, shape):
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def sparse2tensor(mat):
    ''' Converts a Numpy array (W x H) to a sparse Tensor
    '''
    if mat.ndim == 2:
        val = mat.data
        ind = np.vstack((mat.row, mat.col))
        shape = mat.shape

        i = torch.LongTensor(ind)
        v = torch.FloatTensor(val)
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))
    else:
        raise TypeError('mat should have 2 dimensions.')


def normalize3d(img, mean, std):
    ''' Normalizes a voxel Tensor (C x D x H x W) by mean and std. '''
    if len(mean) < 3 or len(std) < 3:
        raise TypeError('not enough means and standard deviations')
    for t, m, s in zip(img, mean, std):
        t.sub_(m).div_(s)
    return img


def tensor2im(image_tensor, imtype=np.uint8):
    ''' Converts a Tensor into a Numpy array.

    Args:
        imtype: the desired type of the converted numpy array
    '''
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:  # if it is greyscale
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    image_numpy = (image_numpy + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def tensor2vid(vid_tensor, vidtype=np.uint8, gray_to_rgb=True):
    ''' Converts a Tensor into a Numpy array but for video. '''
    vid_numpy = vid_tensor[0].cpu().float().numpy()
    if vid_numpy.shape[0] == 1 and gray_to_rgb:
        vid_numpy = np.tile(vid_numpy, (3, 1, 1, 1))
    vid_numpy = np.transpose(vid_numpy, (1, 2, 3, 0))
    vid_numpy = (vid_numpy + 1) / 2.0 * 255.0
    return vid_numpy.astype(vidtype) # change the vidtype to prevent rounding


def rescale_vid(vid, scale=4):
    ''' scikit-video cannot efficiently upscale videos. '''
    T, H, W, C = vid.shape
    new_vid = np.zeros((T, H * scale, W * scale, C))
    for it, ic in zip(range(T), range(C)):
        new_vid[it, :, :, ic] = rescale(vid[it, :, :, ic], scale)
    return new_vid


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print(
            'mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f'
            % (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
