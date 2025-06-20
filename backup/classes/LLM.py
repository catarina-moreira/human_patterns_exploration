import ollama

import re

from classes.ImageData import ImageData

import numpy as np

class LLM:
    def __init__(self, type, llm_model):
        
        if type not in ['ollama', 'openai']:
            raise ValueError("LLM type must be either 'ollama' or 'openai'")
        self.type = type
        self.llm_model = llm_model

    
    def analyse_image(self, image : ImageData, prompt : str, temperature : float = 0.1):

        response = "NA"
        if self.type == 'ollama':
            response = ollama.chat(
                model = self.llm_model,
                messages = [
                    {'role' : 'user',
                     'content' : prompt,
                     'images' : [image.path]
                    }
                ],
                options={"temperature": temperature}  
            )
        
        answer = response.message.content.strip()
        answer = self.remove_numbered_dots(answer)
        total_time = self.ns_to_s(response.total_duration)
        return response, answer, total_time
    
    def ns_to_s(self, ns):
        """Convert nanoseconds to seconds"""
        return np.round(ns / 1e9,2)
    

    def remove_numbered_dots(self, text: str) -> str:
        """
        Removes any numbers followed by a dot from a string.
        
        Args:
            text (str): Input text containing numbered dots
            
        Returns:
            str: Text with numbered dots removed
        """
        text = text.replace("*", "")
        return re.sub(r'\d+\.', '', text)