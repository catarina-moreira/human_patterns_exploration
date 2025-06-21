#!/usr/bin/python
# -*- coding: UTF-8 -*-
from src.core import FixationTask
from typing import List

from src.core.ImageData import ImageData

class PyGazeAnalyzer:

  def __init__(self, fixation_task: FixationTask, screen_dist_cm: float = 60.0, angle_width_deg: float=38.1, 
                angle_height_deg: float=28.6, refresh_rate_hz: float=100, sampling_rate_hz: float=1000, 
                nominal_monitor_inch: float=21.0, target_dpi: int=100):
    
        self.fixations = fixation_task 
        self.screen_dist_cm = screen_dist_cm
        self.angle_width_deg = angle_width_deg
        self.angle_height_deg = angle_height_deg
        self.refresh_rate = refresh_rate_hz
        self.sampling_rate_hz = sampling_rate_hz
        self.nominal_monitor_inch = nominal_monitor_inch
        self.target_dpi = target_dpi
        
        self.image = fixation_task.imageData.image
        self.width = fixation_task.imageData.image.width
        self.height = fixation_task.imageData.image.height

  

