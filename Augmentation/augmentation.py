import torch
import torchvision.transforms.functional as TF
import torchio as tio


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


def verticalFlip(image, label):
    return TF.vflip(image), TF.vflip(label)

def horizontalFlip(image, label):
    return TF.hflip(image), TF.hflip(label)


def rotate(image, label, rotation_angle=30):
    return TF.rotate(image, rotation_angle), TF.rotate(label, rotation_angle)


def translation(image, label,horizontal_shift = 10,vertical_shift = 5):

    # Apply translation transformation
    translated_image = TF.affine(image, angle=0, translate=[horizontal_shift, vertical_shift], scale=1, shear=0)
    translated_label = TF.affine(label, angle=0, translate=[horizontal_shift, vertical_shift], scale=1, shear=0)

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