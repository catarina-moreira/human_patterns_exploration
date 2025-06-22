import os
import time
import base64
import pickle
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod

from src.models.LLM import LLM, OpenAI, Ollama


def format_llm_answer(answer: str) -> str:
    """Format LLM answer as HTML for better display"""
    from IPython.display import display, HTML
    
    sentences = answer.split('. ')
    
    html_output = "<div style='font-family: Arial, sans-serif; line-height: 1.6; font-size: 16px; color: #FFFFFF;'>\n"
    html_output += "  <ul style='margin: 10px 0; padding-left: 20px;'>\n"

    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            html_output += f"    <li>{sentence.capitalize()}.</li>\n"

    html_output += "  </ul>\n"
    html_output += "</div>"

    return display(HTML(html_output))


def create_llm_instance(provider: str = "openai", model: str = None, **kwargs) -> LLM:
    """Factory function to create LLM instances"""
    
    if provider.lower() == "openai":
        model = model or "gpt-4o"
        return OpenAI(model=model, **kwargs)
    elif provider.lower() == "ollama":
        model = model or "llava:34b"
        return Ollama(model=model, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")