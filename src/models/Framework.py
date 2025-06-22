
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


class SceneUnderstandingFramework:
    """Main framework for scene understanding and mask labeling"""
    
    def __init__(self, llm: LLM):
        self.llm = llm
        self.scene_descriptions = {}
        self.mask_labels = {}
        self.processing_log = []
    
    def analyze_scene(self, image_data: ImageData, save_description: bool = True) -> str:
        """Analyze scene and store description"""
        description = self.llm.describe_scene(image_data)
        
        if save_description:
            self.scene_descriptions[image_data.ID] = {
                'description': description,
                'timestamp': time.time(),
                'image_path': image_data.path
            }
        
        self.processing_log.append({
            'action': 'scene_analysis',
            'image_id': image_data.ID,
            'timestamp': time.time(),
            'success': 'Error' not in description
        })
        
        return description
    
    def label_masks_batch(self, masks: List[Mask], image_data: ImageData, 
                         use_context: bool = True, save_results: bool = True) -> Dict[str, str]:
        """Label multiple masks in batch"""
        results = {}
        
        for mask in masks:
            try:
                label = self.llm.label_mask(mask, image_data, use_context)
                results[mask.ID] = label
                
                if save_results:
                    self.mask_labels[mask.ID] = {
                        'label': label,
                        'mask_id': mask.ID,
                        'image_id': image_data.ID,
                        'timestamp': time.time(),
                        'used_context': use_context
                    }
                
                self.processing_log.append({
                    'action': 'mask_labeling',
                    'mask_id': mask.ID,
                    'image_id': image_data.ID,
                    'timestamp': time.time(),
                    'success': 'Error' not in label
                })
                
            except Exception as e:
                print(f"Error labeling mask {mask.ID}: {e}")
                results[mask.ID] = f"Error: {str(e)}"
        
        return results
    
    def save_results(self, output_path: str):
        """Save all results to pickle file"""
        results = {
            'scene_descriptions': self.scene_descriptions,
            'mask_labels': self.mask_labels,
            'processing_log': self.processing_log,
            'llm_stats': {
                'model': self.llm._LLM__model,
                'avg_processing_time': self.llm.get_average_processing_time(),
                'total_requests': len(self.llm.processing_times)
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved to {output_path}")
    
    def load_results(self, input_path: str):
        """Load results from pickle file"""
        with open(input_path, 'rb') as f:
            results = pickle.load(f)
        
        self.scene_descriptions = results.get('scene_descriptions', {})
        self.mask_labels = results.get('mask_labels', {})
        self.processing_log = results.get('processing_log', [])
        
        print(f"Results loaded from {input_path}")
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        total_scenes = len(self.scene_descriptions)
        total_masks = len(self.mask_labels)
        successful_labels = sum(1 for log in self.processing_log 
                               if log['action'] == 'mask_labeling' and log['success'])
        
        return {
            'total_scenes_analyzed': total_scenes,
            'total_masks_labeled': total_masks,
            'successful_labels': successful_labels,
            'success_rate': successful_labels / total_masks if total_masks > 0 else 0,
            'avg_processing_time': self.llm.get_average_processing_time(),
            'total_processing_time': sum(self.llm.processing_times)
        }
