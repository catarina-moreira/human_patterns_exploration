import os

import torch

import numpy as np
import pandas as pd

import pickle

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
import scipy.ndimage as ndimage

import supervision as sv

import pickle

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from src.core import FixationTask
from src.core.ImageData import ImageData
from src.core.Mask import Mask

from src.prompts.gaze_prompts import sample_points_circle, sample_points_cross, sample_points_triangle, sample_points_point

class SAM2:

    def __init__(self, sam_model_path, sam_model_config, task : FixationTask):
        self.sam_model_path = sam_model_path
        self.sam_model_config = sam_model_config
        self.task = task
        self.data = task.data
        self.imageData = task.imageData
        self.image = self.imageData.image
        self.imgID = self.imageData.ID
        self.participant = self.task.participant
        self.partID = self.participant.ID
        self.task.masks[int(self.partID)] = []

        CHECKPOINT = self.sam_model_path
        CONFIG = self.sam_model_config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sam2_model = build_sam2(CONFIG, CHECKPOINT, device=self.device, apply_postprocessing=False)
        print(self.device)
        
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        self.mask_generator = SAM2AutomaticMaskGenerator(self.sam2_model)
        self.predictor.set_image(self.imageData.image)


    def compute_masks_with_prompt(self, ID, prompt : pd.DataFrame, # prompt should be a dataframe
                                prompt_types = ["point", "circle", "triangle", "cross"], 
                                prompt_n_points = 5, prompt_diameter = 10,
                                arm_length_horizontal=11, arm_length_vertical = 11,
                                save_mask=False, output_path=None, size_threshold = 100, DEBUG = False):
        
        mask_preprocessed = {}
        best_mask_found = {}
        participant_data_filtered = prompt
        
        for prompt_type in prompt_types:
            
            if prompt_type == "point":
                print("Processing prompt type:", prompt_type)
                participant_data_filtered = sample_points_point(prompt)
                
            if prompt_type == "circle":
                print("Processing prompt type:", prompt_type)
                participant_data_filtered = sample_points_circle(prompt, N=prompt_n_points, diameter=prompt_diameter)
        
            if prompt_type == "triangle":
                print("Processing prompt type:", prompt_type)
                participant_data_filtered = sample_points_triangle(prompt, arm_length_horizontal=arm_length_horizontal, arm_length_vertical=arm_length_vertical)
            
            if prompt_type == "cross":
                print("Processing prompt type:", prompt_type)
                participant_data_filtered = sample_points_cross(prompt, arm_length_horizontal=arm_length_horizontal, arm_length_vertical=arm_length_vertical)

            if "point" in prompt_type:
                X = [participant_data_filtered.X]
                Y = [participant_data_filtered.Y]
            else:
                X = participant_data_filtered.X.values
                Y = participant_data_filtered.Y.values
                
            aug_prompt = np.array([X,Y], dtype=np.float32)
            aug_prompt = aug_prompt.flatten()
            aug_prompt = aug_prompt.reshape(2, len(X))
            aug_prompt = aug_prompt.T
            label = np.array([1]*len(X), dtype=np.int32)
                
            masks, scores, logits = self.predictor.predict(
                    point_coords=aug_prompt,
                    point_labels=label,
                    multimask_output=True,
            )
            
            # choose the mask with the highest score
            mask_index = np.argmax(scores)
            mask = masks[mask_index]
            score = scores[mask_index]
            score = round(score, 4)
            logits = logits[mask_index]
            
            # Fixed: Initialize mask_preprocessed properly outside the loop
            if prompt_type not in mask_preprocessed:
                mask_preprocessed[prompt_type] = {}

            mask_filename = os.path.join(output_path, prompt_type, f"IMG_{self.imgID}_type_{self.imageData.img_type}_Part_{self.partID}_MASK_{ID}_best_{prompt_type}_mask_{score:.4f}")
            cropped_image_with_alpha, x_min, x_max, y_min, y_max, cropped_mask = self.crop_mask(mask, threshold=0, save_mask=False, output_image_path=mask_filename, size_threshold = size_threshold)
            
            mask_preprocessed[prompt_type]['ID'] = "Img_" + str(self.imageData.ID) + "_Mask_" + str(ID)
            mask_preprocessed[prompt_type]['img_path'] = self.imageData.path
            mask_preprocessed[prompt_type]['mask_path'] = mask_filename
            mask_preprocessed[prompt_type]['mask'] = mask
            mask_preprocessed[prompt_type]['mask_preprocessed'] = self.preprocess_mask(mask)
            mask_preprocessed[prompt_type]['score'] = score
            mask_preprocessed[prompt_type]['logits'] = logits
            mask_preprocessed[prompt_type]['prompt'] = prompt
            mask_preprocessed[prompt_type]['prompt_type'] = prompt_type
            mask_preprocessed[prompt_type]['cropped_image_with_alpha'] = cropped_image_with_alpha
            mask_preprocessed[prompt_type]['x_min'] = x_min
            mask_preprocessed[prompt_type]['x_max'] = x_max
            mask_preprocessed[prompt_type]['y_min'] = y_min
            mask_preprocessed[prompt_type]['y_max'] = y_max
            mask_preprocessed[prompt_type]['cropped_mask'] = cropped_mask
            #mask_preprocessed[prompt_type]['area'] = abs(x_max-x_min) * (y_max-y_min)
            #mask_preprocessed[prompt_type]['perimeter'] = 2 * (abs(x_max-x_min) + abs(y_max-y_min))
        
            if DEBUG:
                self.save_mask_object( Mask(mask_preprocessed[prompt_type]), mask_filename)

        # Find the prompt type that generated the mask with the highest score
        best_prompt_type = None
        best_score = -1
    
        for prompt_type in mask_preprocessed.keys():
            current_score = mask_preprocessed[prompt_type]['score']
            if current_score > best_score:
                best_score = current_score
                best_prompt_type = prompt_type
        
        # Set best_mask_found to the mask_preprocessed with the highest score
        if best_prompt_type is not None:
            best_mask_found = mask_preprocessed[best_prompt_type]
            print(f"Best mask found with prompt type: {best_prompt_type}, score: {best_score:.4f}")
        else:
            # Fallback: if no masks were generated, create empty dict or raise error
            raise ValueError("No masks were generated for any prompt type")

        mask_filename = os.path.join(output_path, "BestMask", f"IMG_{self.imgID}_type_{self.imageData.img_type}_Part_{self.partID}_MASK_{ID}_best_{best_prompt_type}_mask_{best_score:.4f}")
        best_mask_found['mask_path'] = mask_filename + ".png"
    
        final_mask = Mask(best_mask_found)
        self.task.masks[self.partID].append(final_mask)
        if save_mask:
            
            # Fixed: Save the Mask object properly
            self.save_mask_object(final_mask, mask_filename)

        return final_mask
    
    def save_mask(self, mask, cropped_image_with_alpha, mask_filename_path):
        """Save debug outputs including the PNG image with alpha"""
        # get the directory path from the mask_filename_path
        directory_path = os.path.dirname(mask_filename_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        
        # save the RGBA image
        Image.fromarray(mask['cropped_image']).save(mask_filename_path)
        with open(mask_filename_path.replace(".png", ".pkl"), "wb") as f:
            pickle.dump(mask, f)
        Image.fromarray(cropped_image_with_alpha).save(mask_filename_path)
        
    def save_mask_array(self, mask_array, mask_filename_path):
        """Save a numpy array mask"""
        # get the directory path from the mask_filename_path
        directory_path = os.path.dirname(mask_filename_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        
        # Save as pickle
        with open(mask_filename_path + ".pkl", 'wb') as f:
            pickle.dump(mask_array, f)
            
        # Convert to image and save as PNG
        binary_mask_img = (mask_array * 255).astype(np.uint8)
        Image.fromarray(binary_mask_img).save(mask_filename_path + ".png")
        
    def save_mask_object(self, mask_object, mask_filename_path):
        """Save a Mask object"""
        # get the directory path from the mask_filename_path
        directory_path = os.path.dirname(mask_filename_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        
        # Save the complete Mask object as pickle
        with open(mask_filename_path + ".pkl", 'wb') as f:
            pickle.dump(mask_object, f)
        
        # Save the cropped image with alpha as PNG
        Image.fromarray(mask_object.cropped_image_with_alpha).save(mask_filename_path+".png")

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

        if self.imageData.image.shape[:2] != mask.shape:
            raise ValueError("The mask and image must have the same dimensions.")

        y_dim, x_dim, c_dim = self.imageData.image.shape

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
        
        cropped_image = self.imageData.image[scaled_ymin:scaled_ymax, scaled_xmin:scaled_xmax]
        cropped_mask = mask_non_zero[scaled_ymin:scaled_ymax, scaled_xmin:scaled_xmax]

        # Create a new RGBA image from the cropped image
        cropped_image_with_alpha = np.zeros((cropped_image.shape[0], cropped_image.shape[1], 4), dtype=np.uint8)
        cropped_image_with_alpha[..., :3] = cropped_image
        cropped_image_with_alpha[..., 3] = cropped_mask * 255  # Mask to alpha channel conversion

        # save mask to pickle
        if save_mask:
            
            self.save_mask(cropped_image, cropped_image_with_alpha, output_image_path)

            # if not os.path.exists(output_image_path):
            #     os.makedirs(output_image_path)
            # save the RGBA image
            # Image.fromarray(cropped_image).save(output_image_path.replace("image_with_alpha", "mask") + "_thre_" + str(threshold) + ".png")
            # with open(output_image_path + ".pkl", "wb") as f:
            #   pickle.dump(cropped_mask, f)
            #Image.fromarray(cropped_image_with_alpha).save(output_image_path + ".png")

        return cropped_image_with_alpha, x_min, x_max, y_min, y_max, cropped_mask