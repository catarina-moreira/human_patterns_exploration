
import os
import time
import base64
import pickle
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod

import openai
import ollama
import numpy as np
import pandas as pd

from src.core.ImageData import ImageData
from src.core.Mask import Mask
from src.core.Participant import Participant
from src.models.LLM import LLM
from src.core.FixationTask import FixationTask

from src.models.MaskGenerator import SAM2

from src.utils.llm_utils import create_llm_instance


class SceneUnderstandingFramework:
    """Main framework for scene understanding and mask labeling"""
    
    def __init__(self, task : FixationTask, llm_model : str, llm_temp = 0.1 ):

        if "gpt" in llm_model:
            llm_org = "openai"
        else: 
            llm_org = "ollama"

        self.llm_model = llm_model
        self.llm_temp = llm_temp
        self.llm = create_llm_instance(llm_org, llm_model, temperature = llm_temp) 

        self.task = task
        self.image_data = task.imageData
        self.task_data = task.data.copy()
        self.part_id = task.participant.ID
        self.img_id = task.imageData.ID
        self.scene_descriptions = {}
    
    def analyze_scene(self,) -> str:
        """Analyze scene and store description"""
        
        start_time = time.time()
        description = self.llm.describe_scene(self.image_data)
        end_time = time.time()

        self.scene_descriptions[self.img_id] = {
                'description': description,
                'timestamp': end_time - start_time,
                'image_data': self.image_data,
                'llm_model' : self.llm_model,
                'llm_temp' : self.llm_temp
            }
        return description

    def label_mask_batch(self, sam_model_path : str, sam_model_config : str, output_dir = os.path.join("outputs", "Masks"), show_mask = True, max_labels = 10):

        mask_gen = SAM2(sam_model_path = sam_model_path, sam_model_config = sam_model_config, task = task, output_path = output_dir)
        
        scene_description = self.scene_descriptions[self.img_id]['description']
        print("DEBUG ", scene_description)

        for fix_indx in self.task_data.index:

            # select fixation
            fix_prompt = self.task_data.iloc[fix_indx]

            # segment the image in the region that the participant is fixating
            mask = mask_gen.compute_masks_with_prompt(fix_indx, fix_prompt, output_path=output_dir, DEBUG=show_mask, save_mask=True)
            
            if show_mask:
                mask.plot(show=show_mask)

            # generate label candidates for the mask
            mask_labels = []
            for i in range(0, max_labels): 
                mask_label = self.llm.label_mask(mask, self.image_data, use_context = True, context = scene_description) 
                mask_labels.append(mask_label)
            mask.labels = mask_labels

            # save the mask object
            mask_gen.save_mask_object(mask, mask.path)
