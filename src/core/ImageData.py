
from src.core import Mask
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


	def __init__(self, path : str, ID = None, target = None, img_type = None):
		
		self.path = path
		self.ID = ID if ID is not None else self.find_ID()
		self.target = target
		self.width = None
		self.height = None
		self.img_type = img_type
		self.image = self.load()



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
