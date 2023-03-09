import torch
import torchvision.transforms.functional as TF
import nibabel as nib
#import torchio as tio

import numpy as np
from scipy import ndimage
from scipy.ndimage import shift

def verticalFlip(image, label):
    imgvol = np.array( image.dataobj )
    lblvol = np.array( label.dataobj )
    img = np.flipud(imgvol)
    lbl = np.flipud(lblvol)
    image = nib.Nifti1Image ( img, image.affine )
    label = nib.Nifti1Image ( lbl, label.affine )
    return image, label

def horizontalFlip(image, label):
    imgvol = np.array( image.dataobj )
    lblvol = np.array( label.dataobj )
    img = np.fliplr(imgvol)
    lbl = np.fliplr(lblvol)
    image = nib.Nifti1Image ( img, image.affine )
    label = nib.Nifti1Image ( lbl, label.affine )
    return image, label


def rotate(image, label, angle = 30 ):
    #return np.rot90(image), np.rot90(label)
    imgvol = np.array( image.dataobj )
    lblvol = np.array( label.dataobj )
    img = ndimage.rotate(imgvol, angle, reshape=False)
    lbl = ndimage.rotate(lblvol, angle, reshape=False)
    image = nib.Nifti1Image ( img, image.affine )
    label = nib.Nifti1Image ( lbl, label.affine )
    return image, label
   

def addGuassianNoise(image, label, saltPepper = True):
    # Define the standard deviation of the Gaussian noise
    sigma = 0.1

    # Generate Gaussian noise with the same shape as the MRI images
    noise_m = np.random.normal(loc=0, scale=sigma, size= image.shape)
    noise_l = np.random.normal(loc=0, scale=sigma, size= image.shape)

    # Add the noise to the MRI images
    noisy_images = image + noise_m
    noisy_labels = label + noise_l

    return noisy_images, noisy_labels


def changeContrast(image, label, contrast_factor=1.5):
    return TF.adjust_contrast(image, contrast_factor), TF.adjust_contrast( label, contrast_factor)

def addNoise(image, label):
    # Add Gaussian noise
    noisy_image_tensor = image + 0.1*torch.randn_like(image)

    # Add Salt and Pepper noise
    noise = torch.rand_like(image)
    salt = noise > 0.9
    pepper = noise < 0.1
    noisy_image_tensor[salt] = 1.0
    noisy_image_tensor[pepper] = 0.0

    return noisy_image_tensor, label

def elasticFormation(image, label):
    # Define elastic transformation
    elastic = tio.transforms.ElasticDeformation(num_control_points=7, locked_borders=2, image_interpolation='bspline')

    # Apply elastic transformation
    return elastic(image.unsqueeze(0)).squeeze(0), elastic(label.unsqueeze(0)).squeeze(0)





def translation(image, label):

    # shift 10 pixels to the right and 20 pixels down
    shift_amount = (10, 20, 0)  # (shift along height axis, shift along width axis, no shift along channel axis)

    # perform translation
    translated_image = shift(image, shift_amount, cval=0)
    translated_label = shift(label, shift_amount, cval=0)
    
    return translated_image, translated_label


def resize(image, label, scale_factor=0.5):
    scaled_image = TF.resize(image, [int(scale_factor*224), int(scale_factor*224)])
    scaled_label = TF.resize(label, [int(scale_factor*224), int(scale_factor*224)])

    return scaled_image,scaled_label


def crop(image, label, crop_size):
    crop_size = 128

    # Apply random crop transformation
    cropped_image = TF.random_crop(image, [crop_size, crop_size])
    cropped_label = TF.random_crop(label, [crop_size, crop_size])

    return cropped_image, cropped_label