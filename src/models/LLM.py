
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



# =============================================================================
# 1. ENHANCED LLM BASE CLASS
# =============================================================================

class LLM:
    """Enhanced LLM base class with vision capabilities"""
    
    def __init__(self, model: str, results_dir : str = "outputs"):
        self.model = model
        self.scene_context = None
        self.processing_times = []
        self.results_dir = os.path.join(os.getcwd(), results_dir, "SceneDescriptions")
        
        # check if directory exists
        if not os.path.exists(self.results_dir):    
            os.makedirs(self.results_dir, exist_ok=True)
    
    @abstractmethod
    def describe_scene(self, image_data: ImageData, **kwargs) -> str:
        """Generate detailed scene description"""
        pass
    
    @abstractmethod
    def label_mask(self, mask: Mask, image_data: ImageData, use_context: bool = True, **kwargs) -> str:
        """Label a mask based on scene context"""
        pass
    
    def get_average_processing_time(self) -> float:
        """Get average processing time for requests"""
        return np.mean(self.processing_times) if self.processing_times else 0


# =============================================================================
# 2. ENHANCED OPENAI IMPLEMENTATION
# =============================================================================
class OpenAI(LLM):
    """Enhanced OpenAI implementation with vision capabilities"""
    
    def __init__(self, model: str = "gpt-4o", api_key_path: str = "API_Keys/openai.txt", temperature = 0.1):
        super().__init__(model)
        self.api_key_path = api_key_path
        self.client = self.initialize_client()
        self.provider = "OpenAI"
        self.temperature = temperature
        self.results_dir = os.path.join(self.results_dir, "openai_results")

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    
    def initialize_client(self):
        """Initialize OpenAI client with API key"""
        try:
            with open(self.api_key_path, 'r') as file:
                api_key = file.read().strip()
            return openai.OpenAI(api_key=api_key)
        except FileNotFoundError:
            raise FileNotFoundError(f"API key file not found: {self.api_key_path}")
        except Exception as e:
            raise Exception(f"Failed to initialize OpenAI client: {e}")
    
    def encode_image(self, image_path: str) -> str:
        """Encode image as base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def text_query(self, query: str, system_prompt: str = None, **kwargs) -> str:
        """Make a general text query to the LLM"""
        
        if system_prompt is None:
            system_prompt = "You are a helpful AI assistant."
        
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=self.temperature,
                **kwargs
            )
            
            answer = response.choices[0].message.content
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            print(f"Text query completed in {processing_time:.2f} seconds")
            return answer
            
        except Exception as e:
            print(f"Error in text query: {e}")
            return f"Error: {str(e)}"
    
    def query_knowledge_graph(self, triples: List[Tuple[str, str, str]], query: str, **kwargs) -> str:
        """Generate an answer based on knowledge graph triples and a text query"""
        
        # Format the triples into a readable knowledge base
        kg_text = "Knowledge Graph Information:\\n"
        for i, (subject, predicate, object_) in enumerate(triples, 1):
            kg_text += f"{i}. {subject} {predicate} {object_}\\n"
        
        system_prompt = """You are an AI assistant that answers questions based strictly on the provided knowledge graph information. 
        You must only use the facts explicitly stated in the knowledge graph. 
        If the information needed to answer the question is not present in the knowledge graph, clearly state that the information is not available.
        Do not use any external knowledge or make assumptions beyond what is explicitly provided."""
        
        full_query = f"""{kg_text}

                        Question: {query}

                        Please answer the question based only on the information provided in the knowledge graph above."""
        
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_query}
                ],
                temperature=self.temperature,
                **kwargs
            )
            
            answer = response.choices[0].message.content
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            print(f"Knowledge graph query completed in {processing_time:.2f} seconds")

            print(answer)
            return answer
            
        except Exception as e:
            print(f"Error in knowledge graph query: {e}")
            return f"Error: {str(e)}"
    
    def describe_scene(self, image_data: ImageData, custom_prompt: str = None, **kwargs) -> str:
        """Generate detailed scene description using OpenAI GPT-4V"""
        
        if custom_prompt is None:
            scene_prompt = """Analyze the following image and describe its content in detail. 
            Identify **all** objects in the image with their approximate locations.
            Your response should be structured as follows:
            1. **Scene Overview**: General description of the scene type and context.
            2. **Identified Objects**: Comprehensive list of detected objects with their relative positions.
            3. **Spatial Relationships**: How objects relate to each other spatially.
            4. **Scene Context**: The overall purpose or setting of the scene.
            """
        else:
            scene_prompt = custom_prompt
        
        img_base64 = self.encode_image(image_data.path)
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful vision model that analyzes images and provides detailed scene descriptions."},
                    {"role": "user", "content": [
                        {"type": "text", "text": scene_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                    ]}
                ],
                temperature=self.temperature,
                **kwargs
            )
            
            scene_description = response.choices[0].message.content
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            self.scene_context = {
                'description': scene_description,
                'image_id': image_data.ID,
                'timestamp': time.time()
            }
            
            print(f"Scene description completed in {processing_time:.2f} seconds")

            # save the scene_description into a file
            with open(os.path.join(self.results_dir, f"IMG_{image_data.ID}_{image_data.img_type}_descr.txt"), "w") as file:
                file.write(scene_description)

            return scene_description
            
        except Exception as e:
            print(f"Error in scene description: {e}")
            return f"Error: {str(e)}"


    def text_query(self, query: str, system_prompt: str = None, **kwargs) -> str:
        """Make a general text query to the LLM"""
        
        if system_prompt is None:
            system_prompt = "You are a helpful AI assistant."
        
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=self.temperature,
                **kwargs
            )
            
            answer = response.choices[0].message.content
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            print(f"Text query completed in {processing_time:.2f} seconds")
            return answer
            
        except Exception as e:
            print(f"Error in text query: {e}")
            return f"Error: {str(e)}"

    def query_knowledge_graph(self, triples: List[Tuple[str, str, str]], query: str, **kwargs) -> str:
        """Generate an answer based on knowledge graph triples and a text query"""
        
        # Format the triples into a readable knowledge base
        kg_text = "Knowledge Graph Information:\\n"
        for i, (subject, predicate, object_) in enumerate(triples, 1):
            kg_text += f"{i}. {subject} {predicate} {object_}\\n"
        
        system_prompt = """You are an AI assistant that answers questions based strictly on the provided knowledge graph information. 
        You must only use the facts explicitly stated in the knowledge graph. 
        If the information needed to answer the question is not present in the knowledge graph, clearly state that the information is not available.
        Do not use any external knowledge or make assumptions beyond what is explicitly provided."""
        
        full_query = f"""{kg_text}
                Question: {query}
                Please answer the question based only on the information provided in the knowledge graph above."""
        
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_query}
                ],
                temperature=self.temperature,
                **kwargs
            )
            
            answer = response.choices[0].message.content
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            print(f"Knowledge graph query completed in {processing_time:.2f} seconds")
            return answer
            
        except Exception as e:
            print(f"Error in knowledge graph query: {e}")
            return f"Error: {str(e)}"
    
    def label_mask(self, mask: Mask, image_data: ImageData, use_context: bool = True, 
                custom_prompt: str = None, context = None, **kwargs) -> str:
        """Label a mask using OpenAI GPT-4V with optional scene context"""
        
        mask_image_path = mask.path
        
        if custom_prompt is None:
            if use_context and self.scene_context:
                mask_prompt = f"""You previously analyzed the main scene and described it as follows:
                {context}
                
                Now, analyze the new image, which is a **masked portion** from the main scene.
                Your task:
                1. Identify the primary object in the masked region.
                2. Assign the most contextually appropriate label based on the scene analysis.
                3. Consider the object's role and function in the scene.
                4. Provide a single, descriptive word as the label.
                
                Respond with just the object name (one word).
                """
            else: # no context
                mask_prompt = """Analyze this image showing a masked portion from a larger scene.
                Your task:
                1. Identify the primary object in the masked region.
                2. Provide the most appropriate single-word label.
                3. Consider the object's apparent function and characteristics.
                
                Respond with just the object name (one word).
                """
        else:
            mask_prompt = custom_prompt
        
        img_base64 = self.encode_image(mask_image_path)
        
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful vision model that labels objects in images with contextual understanding."},
                    {"role": "user", "content": [
                        {"type": "text", "text": mask_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                    ]}
                ],
                temperature=self.temperature,
                **kwargs
            )
            
            label = response.choices[0].message.content.strip()
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            print(f"Mask labeling completed in {processing_time:.2f} seconds")
            return self.clean_label(label)
            
        except Exception as e:
            print(f"Error in mask labeling: {e}")
            return f"Error: {str(e)}"
    
    def clean_label(self, label: str) -> str:
        """Clean and standardize the label output"""
        # Remove common prefixes/suffixes
        label = label.replace("[", "").replace("]", "")
        label = label.replace("Object Name:", "").replace("- ", "")
        label = label.strip().lower()
        
        # Take first word if multiple words
        if " " in label:
            label = label.split()[0]
        
        return label

# =============================================================================
# 3. ENHANCED OLLAMA IMPLEMENTATION
# =============================================================================

class Ollama(LLM):
    """Enhanced Ollama implementation with vision capabilities"""
    
    def __init__(self, model: str = "llava:34b", temperature: float = 0.1):
        super().__init__(model)
        self.test_connection()
        self.temperature = temperature
        self.llm_provider = "Ollama"
        self.results_dir = os.path.join(self.results_dir, "ollama_results")

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def test_connection(self):
        """Test Ollama connection"""
        try:
            ollama.list()
            print(f"Ollama connection successful. Using model: {self.model}")
        except Exception as e:
            print(f"Warning: Ollama connection failed: {e}")
    
    def describe_scene(self, image_data: ImageData, custom_prompt: str = None, **kwargs) -> str:
        """Generate detailed scene description using Ollama"""
        
        if custom_prompt is None:
            scene_prompt = """Analyze the following image and describe its content in detail.
            Identify **all** objects in the image with their approximate locations.
            Your response should be structured as follows:
            1. **Scene Overview**: General description of the scene type and context.
            2. **Identified Objects**: Comprehensive list of detected objects with their relative positions.
            3. **Spatial Relationships**: How objects relate to each other spatially.
            4. **Scene Context**: The overall purpose or setting of the scene.
            """
        else:
            scene_prompt = custom_prompt
        
        start_time = time.time()
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful vision model that analyzes images and provides detailed scene descriptions."},
                    {"role": "user", "content": scene_prompt, "images": [image_data.path]}
                ],
                **kwargs
            )
            
            scene_description = response['message']['content']
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Store context for future mask labeling
            self.scene_context = {
                'description': scene_description,
                'image_id': image_data.ID,
                'timestamp': time.time()
            }

            with open(os.path.join(self.results_dir, f"IMG_{image_data.ID}_{image_data.img_type}_ descr.txt"), "w") as file:
                file.write(scene_description)

            
            print(f"Scene description completed in {processing_time:.2f} seconds")
            return scene_description
            
        except Exception as e:
            print(f"Error in scene description: {e}")
            return f"Error: {str(e)}"
    
    def label_mask(self, mask: Mask, image_data: ImageData, use_context: bool = True, 
                   custom_prompt: str = None, **kwargs) -> str:
        """Label a mask using Ollama with optional scene context"""
        
        # Get mask image path
        if hasattr(mask, 'cropped_image_path'):
            mask_image_path = mask.cropped_image_path
        else:
            # Fallback: assume mask has been saved with standard naming
            mask_image_path = f"temp_mask_{mask.ID}.png"
            if not os.path.exists(mask_image_path):
                raise FileNotFoundError(f"Mask image not found: {mask_image_path}")
        
        if custom_prompt is None:
            if use_context and self.scene_context:
                mask_prompt = f"""You previously analyzed the main scene and described it as follows:
                {self.scene_context['description']}
                
                Now, analyze the new image, which is a **masked portion** from the main scene.
                Your task:
                1. Identify the primary object in the masked region.
                2. Assign the most contextually appropriate label based on the scene analysis.
                3. Consider the object's role and function in the scene.
                4. Provide a single, descriptive word as the label.
                
                Respond with just the object name (one word).
                """
            else:
                mask_prompt = """Analyze this image showing a masked portion from a larger scene.
                Your task:
                1. Identify the primary object in the masked region.
                2. Provide the most appropriate single-word label.
                3. Consider the object's apparent function and characteristics.
                
                Respond with just the object name (one word).
                """
        else:
            mask_prompt = custom_prompt
        
        start_time = time.time()
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful vision model that labels objects in images with contextual understanding."},
                    {"role": "user", "content": mask_prompt, "images": [mask_image_path]}
                ],
                **kwargs
            )
            
            label = response['message']['content'].strip()
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            print(f"Mask labeling completed in {processing_time:.2f} seconds")
            return self.clean_label(label)
            
        except Exception as e:
            print(f"Error in mask labeling: {e}")
            return f"Error: {str(e)}"
    
    def clean_label(self, label: str) -> str:
        """Clean and standardize the label output"""
        # Remove common prefixes/suffixes
        label = label.replace("[", "").replace("]", "")
        label = label.replace("Object Name:", "").replace("- ", "")
        label = label.strip().lower()
        
        # Take first word if multiple words
        if " " in label:
            label = label.split()[0]
        
        return label

