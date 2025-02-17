
from classes.ImageData import ImageData

import scipy.ndimage as ndimage

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle

import supervision as sv


class Mask:

    def __init__(self, mask_dict):
        
        self.mask = mask_dict['mask']

        self.img_path = mask_dict['img_path']
        self.ID = mask_dict['ID']
        self.score = mask_dict['score']
        self.logits = mask_dict['logits']
        self.prompt = mask_dict['prompt']
        self.prompt_type = mask_dict['prompt_type']
        self.cropped_image_with_alpha = mask_dict['cropped_image_with_alpha']
        self.x_min = mask_dict['x_min']
        self.x_max = mask_dict['x_max']
        self.y_min = mask_dict['y_min']
        self.y_max = mask_dict['y_max']
        self.cropped_mask = mask_dict['cropped_mask']
        self.area = abs(self.x_max-self.x_min) * (self.y_max-self.y_min)
        self.perimeter = 2 * (abs(self.x_max-self.x_min) + abs(self.y_max-self.y_min))
        self.path = ""
        self.labels = []

    def __str__(self):
        return (f"Mask(ID='{self.ID}', Score={self.score:.2f}, "
                f"Prompt='{self.prompt}', Area={self.area}, Perimeter={self.perimeter}, "
                f"BoundingBox=({self.x_min}, {self.y_min}) -> ({self.x_max}, {self.y_max}), "
                f"Path='{self.path}')")


    def plot_mask(self, figsize=(2,2)):

        sv.plot_images_grid(
                images=[self.cropped_mask],
                grid_size=(1, 1),
                size = figsize)


    def plot(self, figsize=(2,2), title=None):
        """Display the image"""
        plt.grid(False)
        plt.axis('off')
        plt.gcf().set_size_inches(figsize[0],figsize[1])
        plt.imshow(self.cropped_image_with_alpha)

        if title is not None:
            plt.title(title)

        plt.show()
        
    def plot_cropped_mask(self, figsize=(2,2), title=None):
        """Display the image"""
        plt.grid(False)
        plt.axis('off')
        plt.gcf().set_size_inches(figsize[0],figsize[1])
        plt.imshow(self.cropped_mask)

        if title is not None:
            plt.title(title)

        plt.show()

    