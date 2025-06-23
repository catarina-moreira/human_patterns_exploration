from typing import List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os

import math

import re

import matplotlib.patheffects as path_effects


class ImageData(object):


	def __init__(self, path : str, ID = None, target : List = None, img_type : str = None):
		
		self.path = path
		self.ID = ID if ID is not None else self.find_ID()
		self.target = target
		self.width = None
		self.height = None
		self.img_type = img_type
		self.image = self.load()
		self.target = target

	def find_ID(self):
		self.ID = os.path.splitext(os.path.basename(self.path))[0]
		# if exists, extract the integer in the filename and use it as the ID. 
		# Otherwise, use the filename as ID
		match = re.search(r'\d+', self.ID)
		self.ID = int(match.group()) if match else self.ID
		return self.ID

	def load(self):
		if not os.path.isfile(self.path):
			raise FileNotFoundError(f"Image not found: {self.path}")

		image = cv2.imread(self.path, cv2.IMREAD_COLOR)
		if image is None:
			raise ValueError("Could not load the image with cv2")

		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		self.height, self.width, _ = image.shape
		return image

	def show(self, figsize=(12,8), dpi = 300, title = None):
			plt.grid(False)
			plt.axis('off')
			plt.gcf().set_size_inches(figsize[0],figsize[1])
			plt.gcf().set_dpi(dpi)

			if not title:
				plt.title(f"Image {self.ID}") 
			else: 
				plt.title(title) 
			

			plt.imshow(self.image)

	def highlight_target(self, color=(255, 0, 0), thickness=3, show=True, figsize=(12,8), dpi=300, title=None):
		"""
		Highlights a bounding box on the image at the specified coordinates.
		
		Parameters:
		- x1, x2: X coordinates (x1 should be left, x2 should be right)
		- y1, y2: Y coordinates (y1 should be top, y2 should be bottom)
		- color: RGB color tuple for the bounding box (default: red)
		- thickness: Thickness of the bounding box lines (default: 3)
		- show: Whether to display the image (default: True)
		- figsize: Figure size for display (default: (12,8))
		- dpi: DPI for display (default: 300)
		- title: Title for the plot (default: None)
		
		Returns:
		- numpy array: Image with highlighted bounding box
		"""
		# Create a copy of the image to avoid modifying the original
		highlighted_image = self.image.copy()
		
		x1 = self.target[0]
		x2 = self.target[1]
		y1 = self.target[2]
		y2 = self.target[3]

		# Ensure coordinates are integers and in correct order
		x1, x2 = int(min(x1, x2)), int(max(x1, x2))
		y1, y2 = int(min(y1, y2)), int(max(y1, y2))
		
		# Clamp coordinates to image boundaries
		x1 = max(0, min(x1, self.width - 1))
		x2 = max(0, min(x2, self.width - 1))
		y1 = max(0, min(y1, self.height - 1))
		y2 = max(0, min(y2, self.height - 1))
		
		# Draw the bounding box rectangle
		cv2.rectangle(highlighted_image, (x1, y1), (x2, y2), color, thickness)
		
		# Display the image if requested
		if show:
			plt.figure()
			plt.grid(False)
			plt.axis('off')
			plt.gcf().set_size_inches(figsize[0], figsize[1])
			plt.gcf().set_dpi(dpi)
			
			if not title:
				plt.title(f"Image {self.ID} - Highlighted Target")
			else:
				plt.title(title)
			
			plt.imshow(highlighted_image)
			plt.show()
		
		return highlighted_image