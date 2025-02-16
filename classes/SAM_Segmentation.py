import os

import torch

import numpy as np

import pickle

import matplotlib.pyplot as plt

HOME = os.getcwd()  

from PIL import Image, ImageDraw
import scipy.ndimage as ndimage

os.chdir(os.path.join(HOME, "segment-anything-2"))

import supervision as sv

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from classes.ImageData import ImageData
from classes.Mask import Mask

os.chdir(HOME)

class SAM_Segmentation:

    def __init__(self, sam_model_path, sam_model_config, image : ImageData, part_ID):
        self.sam_model_path = sam_model_path
        self.sam_model_config = sam_model_config
        self.image_data = image
        self.part_ID = part_ID
        self.image_data.masks[int(self.part_ID)] = []

        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        CHECKPOINT = self.sam_model_path
        CONFIG = self.sam_model_config

        self.device = DEVICE
        self.sam2_model = build_sam2(CONFIG, CHECKPOINT, device=DEVICE, apply_postprocessing=False)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        self.mask_generator = SAM2AutomaticMaskGenerator(self.sam2_model)

        self.predictor.set_image(self.image_data.image)


    def compute_masks_with_prompt(self, X ,Y, ID, save_mask=False, output_image_path=None, size_threshold = 100):

        prompt = np.array([X,Y], dtype=np.float32)
        prompt = prompt.flatten()
        prompt = prompt.reshape(2, len(X))
        prompt = prompt.T
        label = np.array([1]*len(X), dtype=np.int32)
            
        masks, scores, logits = self.predictor.predict(
                point_coords=prompt,
                point_labels=label,
                multimask_output=True,
        )
        
        # choose the mask with the highest score
        mask_index = np.argmax(scores)
        mask = masks[mask_index]
        score = scores[mask_index]
        logits = logits[mask_index]

        cropped_image_with_alpha, x_min, x_max, y_min, y_max, cropped_mask = self.crop_mask(mask, threshold=0, save_mask=save_mask, output_image_path=output_image_path, size_threshold = size_threshold)

        mask_preprocessed = {}

        mask_preprocessed['ID'] = "Img_" + self.image_data.ID + "_Mask_" + str(ID)
        mask_preprocessed['img_path'] = self.image_data.path
        mask_preprocessed['mask'] = mask
        mask_preprocessed['mask_preprocessed'] = self.preprocess_mask(mask)
        mask_preprocessed['score'] = score
        mask_preprocessed['logits'] = logits
        mask_preprocessed['prompt'] = prompt
        mask_preprocessed['cropped_image_with_alpha'] = cropped_image_with_alpha
        mask_preprocessed['x_min'] = x_min
        mask_preprocessed['x_max'] = x_max
        mask_preprocessed['y_min'] = y_min
        mask_preprocessed['y_max'] = y_max
        mask_preprocessed['cropped_mask'] = cropped_mask
        mask_preprocessed['area'] = abs(x_max-x_min) * (y_max-y_min)
        mask_preprocessed['perimeter'] = 2 * (abs(x_max-x_min) + abs(y_max-y_min))

        final_mask = Mask(mask_preprocessed)

        self.image_data.masks[self.part_ID].append(final_mask)

        return final_mask
    
    def preprocess_mask(self, mask, size_threshold=100):

        # Label connected components
        labeled_mask, num_features = ndimage.label(mask)

        # Create a new mask where small components are removed
        component_sizes = np.bincount(labeled_mask.ravel())
        too_small = component_sizes < size_threshold
        too_small_mask = too_small[labeled_mask]

        # Zero out small components
        filtered_mask = mask.copy()
        filtered_mask[too_small_mask] = 0

        return filtered_mask
    
    def crop_mask(self, mask, threshold =0, save_mask=False, output_image_path=None, size_threshold = 100):

        if self.image_data.image.shape[:2] != mask.shape:
            raise ValueError("The mask and image must have the same dimensions.")

        y_dim, x_dim, c_dim = self.image_data.image.shape

        mask_non_zero = mask > 0
        mask_non_zero = self.preprocess_mask( mask_non_zero, size_threshold = size_threshold )
        coords = np.argwhere(mask_non_zero)

        if len(coords) == 0:
            mask_non_zero = mask > 0
            coords = np.argwhere(mask_non_zero)

        y_min, x_min = np.min(coords, axis=0)
        y_max, x_max = np.max(coords, axis=0)

        # Crop the image to the bounding box
        scaled_ymin = int(y_min - threshold if y_min - threshold > 0 else y_min)
        scaled_xmin = int(x_min - threshold if x_min - threshold > 0 else x_min)
        scaled_ymax = int(y_max + threshold if y_max + threshold + 1 < y_dim else y_max + 1)
        scaled_xmax = int(x_max + threshold if x_max + threshold + 1 < x_dim else x_max + 1)
        
        cropped_image = self.image_data.image[scaled_ymin:scaled_ymax, scaled_xmin:scaled_xmax]
        cropped_mask = mask_non_zero[scaled_ymin:scaled_ymax, scaled_xmin:scaled_xmax]

        # Create a new RGBA image from the cropped image
        cropped_image_with_alpha = np.zeros((cropped_image.shape[0], cropped_image.shape[1], 4), dtype=np.uint8)
        cropped_image_with_alpha[..., :3] = cropped_image
        cropped_image_with_alpha[..., 3] = cropped_mask * 255  # Mask to alpha channel conversion

        # save mask to pickle
        if save_mask:
            # save the RGBA image
            Image.fromarray(cropped_image).save(output_image_path.replace("image_with_alpha", "mask") + "_thre_" + str(threshold) + ".png")
            with open(output_image_path + ".pkl", "wb") as f:
                pickle.dump(cropped_mask, f)
            Image.fromarray(cropped_image_with_alpha).save(output_image_path + ".png")

        return cropped_image_with_alpha, x_min, x_max, y_min, y_max, cropped_mask
    
    


