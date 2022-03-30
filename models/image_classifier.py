'''
TODO: Build a image loader
TODO: Load a pre-built model and train with triplet loss
'''
import random
import cv2
import os
from torchvision import datasets, transforms
from PIL import Image, ImageOps
import numpy as np
from scipy import ndimage
import torch
import tifffile as tiffreader


class ReturnLabeledMyxo(datasets.ImageFolder):
    def __init__(self, root, crop_x, crop_y, resize=0.25, norm_max=1, norm_min=-1, transform=None, target_transform=None,
                 loader=tiffreader.TiffFile, has_mask=False, mask_dir=None,
                 pre_load=False, is_he=False, is_training=True, same_label=True):
        """
        This function can return an image in the Myxo dataset with label
        :param root (string): Root directory path.
        :param crop_x:
        :param crop_y:
        :param resize:
        :param norm_min:
        :param norm_max:
        :param transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        :param target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        :param loader (callable, optional): A function to load an image given its path.
        :param pre_load: Pre load the images to memory
        :param is_he: Use histogram equalization when pre_load
        :param is_training: Is training mode
        :param same_label: Do 2 images have the same label
        """
        super().__init__(root, transform, target_transform, loader)
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.resize = resize
        self.norm_max = norm_max
        self.norm_min = norm_min
        self.crop_diag = np.sqrt(np.square(crop_x) + np.square(crop_y))
        self.pre_load = pre_load
        self.is_he = is_he
        self.is_training = is_training
        self.same_label = same_label
        self.has_mask = has_mask
        self.mask_dir = mask_dir
        if self.pre_load:
            self.preload_images = []
            for (fp, idx) in self.samples:
                img = self.loader(fp).asarray()
                if self.is_he:
                    img = cv2.equalizeHist(img)  # ImageOps.equalize(img)
                ''' The resized shape is the transpose of np.shape() '''
                img_size = np.array([np.shape(img)[1] * self.resize, np.shape(img)[0] * self.resize], dtype=int)
                if self.has_mask and self.mask_dir:
                    img_dir_chain = fp.split("/")
                    mask = self.loader(os.path.join(self.mask_dir, img_dir_chain[-2], img_dir_chain[-1])).asarray()
                    img = np.array([img, mask, 255 - mask]).transpose((1, 2, 0))
                img = cv2.resize(img, tuple(img_size), interpolation=cv2.INTER_LANCZOS4)
                self.preload_images.append(img)
        self.targets = np.array(self.targets)

    def rotate_images(self, images, angle):
        # Get the image size
        image = images[0]
        image_size = image.shape
        new_len = np.max([abs(self.crop_diag * np.cos((angle + 45) / 180 * np.pi)),
                         abs(self.crop_diag * np.sin((angle + 45) / 180 * np.pi))])
        new_len = int(np.ceil(new_len))
        x_sample = np.random.randint(new_len//2, image_size[0] - np.ceil(new_len/2))
        y_sample = np.random.randint(new_len//2, image_size[1] - np.ceil(new_len/2))

        x_min = min([x_sample, 0])
        y_min = min([y_sample, 0])

        rot_imgs = [ndimage.rotate(img[x_min:x_min+new_len, y_min:y_min+new_len], angle, reshape=True) for img in images]
        rot_img_shape = rot_imgs[0].shape
        img_center = [rot_img_shape[0] // 2, rot_img_shape[1] // 2]
        return [img[img_center[0] - self.crop_x // 2:img_center[0] + self.crop_x // 2,
                img_center[1] - self.crop_y // 2:img_center[1] + self.crop_y // 2] for img in rot_imgs]


    def __getitem__(self, index):
        if self.pre_load:
            img0 = self.preload_images[index]
        else:
            img0 = self.loader(self.samples[index][0]).asarray()
            ''' The resized shape is the transpose of np.shape() '''
            img_size = np.array([np.shape(img0)[1] * self.resize, np.shape(img0)[0] * self.resize], dtype=int)
            if self.has_mask and self.mask_dir:
                img_dir_chain = self.samples[index][0].split("/")
                mask = self.loader(os.path.join(self.mask_dir, img_dir_chain[-2], img_dir_chain[-1])).asarray()
                img0 = np.array([img0, mask, 255 - mask]).transpose((1, 2, 0))
            img0 = cv2.resize(img0, tuple(img_size), interpolation=cv2.INTER_LANCZOS4)

        target = self.targets[index]
        if self.same_label:
            class_idxs = np.where(self.targets == target)[0]
            img1_idx = np.random.randint(len(class_idxs))
        else:
            class_idxs = np.where(self.targets != target)[0]
            img1_idx = np.random.randint(len(class_idxs))

        if self.pre_load:
            img1 = self.preload_images[class_idxs[img1_idx]]
        else:
            img1 = self.loader(self.samples[class_idxs[img1_idx]][0]).asarray()
            ''' The resized shape is the transpose of np.shape() '''
            img_size = np.array([np.shape(img1)[1] * self.resize, np.shape(img1)[0] * self.resize], dtype=int)
            img1 = cv2.resize(img1, tuple(img_size), interpolation=cv2.INTER_LANCZOS4)
            if self.has_mask and self.mask_dir:
                img_dir_chain = self.samples[class_idxs[img1_idx]][0].split("/")
                mask = self.loader(os.path.join(self.mask_dir, img_dir_chain[-2], img_dir_chain[-1])).asarray()
                img1 = np.array([img1, mask, 255 - mask]).transpose((1, 2, 0))
            img1 = cv2.resize(img1, tuple(img_size), interpolation=cv2.INTER_LANCZOS4)

            if self.is_he:
                img0 = cv2.equalizeHist(img0)
                img1 = cv2.equalizeHist(img1)

        # rotate and crop
        if self.is_training:
            [img0] = self.rotate_images([img0], angle=float(np.random.random(1) * 360 - 180))
            [img1] = self.rotate_images([img1], angle=float(np.random.random(1) * 360 - 180))
            if random.random() < 0.5:
                img0 = np.fliplr(img0)
            if random.random() < 0.5:
                img1 = np.fliplr(img1)
        else:
            img_shape = np.shape(img0)
            img_center = [img_shape[0] // 2, img_shape[1] // 2]
            img0 = img0[img_center[0] - self.crop_x // 2:img_center[0] + self.crop_x // 2,
                        img_center[1] - self.crop_y // 2:img_center[1] + self.crop_y // 2]
            img1 = img1[img_center[0] - self.crop_x // 2:img_center[0] + self.crop_x // 2,
                        img_center[1] - self.crop_y // 2:img_center[1] + self.crop_y // 2]


        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        if not self.has_mask:
            img0 = img0[np.newaxis, :, :]
            img1 = img1[np.newaxis, :, :]
        else:
            img0 = img0.transpose((2, 0, 1))
            img1 = img1.transpose((2, 0, 1))
        # Normalize image to normal distribution
        #img0 = img0 / np.iinfo(img0.dtype).max * (self.norm_max - self.norm_min) + self.norm_min
        #img1 = img1 / np.iinfo(img1.dtype).max * (self.norm_max - self.norm_min) + self.norm_min
        if self.has_mask:
            img0 = np.array([(img0[0] - np.mean(img0[0])) / np.std(img0[0]), img0[1] / 255 * 2 - 1, img0[2] / 255 * 2 - 1])
            img1 = np.array([(img1[0] - np.mean(img1[0])) / np.std(img1[0]), img1[1] / 255 * 2 - 1, img1[2] / 255 * 2 - 1])
        else:
            img0[0] = (img0[0] - np.mean(img0[0])) / np.std(img0[0])
            img1[0] = (img1[0] - np.mean(img1[0])) / np.std(img1[0])

        img0 = torch.FloatTensor(img0.copy())
        img1 = torch.FloatTensor(img1.copy())

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img0, img1), self.classes[target]