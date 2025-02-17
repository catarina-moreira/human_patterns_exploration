
import os
HOME = os.getcwd()

import openai
import ollama

import time
import base64

import numpy as np

from classes.ImageData import ImageData

import pandas as pd

from IPython.display import display, HTML


from utils.prompts import *

import pickle


def format_llm_answer(answer: str) -> str:
        """
        Formats a text answer from an LLM into a user-friendly HTML format.

        Args:
            answer (str): The raw text from the LLM.

        Returns:
            str: A formatted HTML version of the answer.
        """
        # Split the text into sentences
        sentences = answer.split('. ')
        
        # Start creating the HTML structure
        html_output = "<div style='font-family: Arial, sans-serif; line-height: 1.6; font-size: 16px; color: #FFFFFF;'>\n"
        html_output += "  <ul style='margin: 10px 0; padding-left: 20px;'>\n"

        # Add each sentence as a bullet point
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:  # Skip empty sentences
                html_output += f"    <li>{sentence.capitalize()}.</li>\n"

        # Close the HTML structure
        html_output += "  </ul>\n"
        html_output += "</div>"

        return display(HTML(html_output))

# Function to encode image as base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def open_ai_scene_description(scene_path, model="gpt-4o", temperature=0.1):
    with open('API_Keys/openai.txt', 'r') as file:
        OPENAI_API_KEY = file.read().strip()

    # Load image data
    img_scene_base64 = encode_image(scene_path.path)


    # Initialize OpenAI client
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # Step 1: Describe the main scene
    scene_prompt = """Analyze the following image and describe its content in detail. 
    Identify **all** objects in the image.
    Your response should be structured as follows:
    1. **Scene Overview**: General description of the scene.
    2. **Identified Objects**: List of detected objects with their relative positions.
    """

    start_time = time.time()

    scene_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful vision model that analyzes images and provides detailed scene descriptions."},
            {"role": "user", "content": [
                {"type": "text", "text": scene_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_scene_base64}"}}
            ]}
        ],
        temperature=temperature
    )

    scene_desc = scene_response.choices[0].message.content
    print("Main Scene Processed in:", time.time() - start_time, "sec")
    return scene_desc


def ollama_scene_description(img_scene_data, model: str = "llava:34b"):

    scene_prompt = """Analyze the following image and describe its content in detail.
    Identify **all** objects in the image.
    Your response should be structured as follows:
    1. **Scene Overview**: General description of the scene.
    2. **Identified Objects**: List of detected objects with their relative positions.
    """

    start_time = time.time()
    res = ollama.chat(
        model=model,
        messages=[
                {"role": "system", "content": "You are a helpful vision model that analyzes images and provides detailed scene descriptions."},
                {"role": "user", 'content': scene_prompt,
                'images': [img_scene_data.path]
            }
        ]
    )
    scene_desc = res['message']['content']

    print("Main Scene Processed in:", time.time() - start_time, "sec")
    
    return scene_desc

def ollama_mask_labellinh_with_context(scene_desc, img_mask_data, model: str = "llava:34b"):

    mask_prompt = f"""You previously analyzed the main scene and described it as follows:
    {scene_desc}
    Now, analyze the new image, which is a **subset** (masked portion) of the main scene.
    Your task:
    1. Identify the object in the masked region.
    2. Assign the most contextually appropriate label based on the previous scene analysis.
    3. Your label should be a single word.

    Respond with:
    - [Object Name]
    """

    start_time = time.time()
    res = ollama.chat(
        model=model,
        messages=[
                {"role": "system", "content": "You are a helpful vision model that labels masked portions of an image based on the scene context."},
                {"role": "user", 'content': [ 
                    mask_prompt,
                    {'images': [img_mask_data.path]}
                ]
            }
        ]
    )
    scene_desc = res['message']['content']

    print("Main Scene Processed in:", time.time() - start_time, "sec")
    
    return scene_desc


def open_ai_mask_labeling_with_context(scene_desc, img_mask_data, temperature=0.1):
        
    with open('API_Keys/openai.txt', 'r') as file:
        OPENAI_API_KEY = file.read().strip()

    img_mask_base64 = encode_image(img_mask_data.path)

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # Step 2: Label the mask based on the scene context
    mask_prompt = f"""You previously analyzed the main scene and described it as follows:
    {scene_desc}
    Now, analyze the new image, which is a **subset** (masked portion) of the main scene.
    Your task:
    1. Identify the object in the masked region.
    2. Assign the most contextually appropriate label based on the previous scene analysis.
    3. Your label should be a single word.

    Respond with:
    - [Object Name]
    """

    start_time = time.time()

    mask_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful vision model that labels masked portions of an image based on the scene context."},
            {"role": "user", "content": [
                {"type": "text", "text": mask_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_mask_base64}"}}
            ]}
        ],
        temperature=temperature
    )

    mask_label = mask_response.choices[0].message.content
    print("Mask Processed in:", time.time() - start_time, "sec")

    return mask_label



def open_ai_mask_labeling(img_mask_data, temperature=0.1):
        
    with open('API_Keys/openai.txt', 'r') as file:
        OPENAI_API_KEY = file.read().strip()

    img_mask_base64 = encode_image(img_mask_data.path)

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # Step 2: Label the mask based on the scene context
    mask_prompt = f"""Analyze the image, which is a **subset** (masked portion) of a main scene.
    Your task:
    1. Identify the object in the masked region.
    2. Assign the most contextually appropriate label based on the previous scene analysis.
    3. Your label should be a single word.

    Respond with:
    - [Object Name]
    """

    start_time = time.time()

    mask_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful vision model that labels masked portions of an image based on the scene context."},
            {"role": "user", "content": [
                {"type": "text", "text": mask_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_mask_base64}"}}
            ]}
        ],
        temperature=temperature
    )

    mask_label = mask_response.choices[0].message.content
    print("Mask Processed in:", time.time() - start_time, "sec")

    return mask_label
