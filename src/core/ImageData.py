from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os

from src.core.Mask import Mask

import math
import pickle

import re

import matplotlib.patheffects as path_effects
import matplotlib.patches as patches

import Levenshtein

import nltk

print("Downloading required NLTK data...")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng') 
nltk.download('wordnet')
nltk.download('omw-1.4') 
print("NLTK data downloaded successfully!")

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

class ImageData(object):

	def __init__(self, path : str, ID = None, target : List = None, img_type : str = None, results_dir = os.path.join(".", "outputs", "SceneDescriptions", "openai_results")):
		
		self.path = path
		self.ID = ID if ID is not None else self.find_ID()
		self.target = target
		self.width = None
		self.height = None
		self.img_type = img_type
		self.image = self.load()
		self.target = target
		self.results_dir = results_dir
		self.description = self.load_scene_description()	
		self.masks = []
		self.mask_labels = None

	def find_ID(self):
		self.ID = os.path.splitext(os.path.basename(self.path))[0]
		# if exists, extract the integer in the filename and use it as the ID. 
		# Otherwise, use the filename as ID
		match = re.search(r'\d+', self.ID)
		self.ID = int(match.group()) if match else self.ID
		return self.ID

	def load_scene_description(self):

		scene_filename = f"IMG_{self.ID}_{self.img_type}_descr.txt"
		description = None
		try:
			with open(os.path.join(self.results_dir, scene_filename), 'r') as file:
				description = file.read()
			return description.strip()
		except FileNotFoundError as e:
			
			print(f"[WARNING] Scene description file not found. Please run LLM.describe_scene")
			print(f"File path: {os.path.join(self.results_dir, scene_filename)}")
			return description


	def load_img_masks(self, masks_path_dir, part_id = None):

		mask_files = os.listdir(masks_path_dir)
		mask_files = [file for file in mask_files if file.endswith('.pkl')]
		if part_id is not None:
			mask_files = [file for file in mask_files if f"IMG_{self.ID}_type_{self.img_type}_Part_{part_id}_" in file]
		else:
			mask_files = [file for file in mask_files if f"IMG_{self.ID}_type_{self.img_type}" in file]

		for file in mask_files:
			try:
				with open(os.path.join(masks_path_dir, file), 'rb') as mask_file:
					mask_data = pickle.load(mask_file)
			except Exception as e:
				print("[ERROR] Mask file ", os.path.join(masks_path_dir, file), "could not be loaded")
				return None
					
			mask_data.get_most_frequent_label()
			self.masks.append(mask_data)
		
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


	def plot_mask_labels(self, figsize=(12, 8), dpi=300, title=None, 
               line_thickness=2, font_size=10, show_labels=True, alpha=0.7):

		bboxes_dict = self.mask_labels
		if bboxes_dict is None or len(bboxes_dict) == 0:
			print("You did not compute the unique mask labels. Run ImageData.get_scene_masks()")
			return

		def _check_label_overlap(pos1, pos2, text1, text2, font_size):
			"""Check if two label positions would overlap"""
			# Approximate text dimensions (rough estimation)
			char_width = font_size * 0.6
			char_height = font_size * 1.2
			
			w1, h1 = len(text1) * char_width, char_height
			w2, h2 = len(text2) * char_width, char_height
			
			x1, y1 = pos1
			x2, y2 = pos2
			
			# Check if rectangles overlap
			return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)
		
		def _find_best_label_position(bbox, object_name, existing_positions, font_size, img_width, img_height):
			"""Find the best position for a label to avoid overlaps"""
			x_min, x_max, y_min, y_max = bbox
			
			# Define possible positions around the bounding box
			positions = [
				(x_min, y_min - 5),           # Top-left (default)
				(x_min, y_max + 15),          # Bottom-left
				(x_max - len(object_name) * font_size * 0.6, y_min - 5),  # Top-right
				(x_max - len(object_name) * font_size * 0.6, y_max + 15), # Bottom-right
				(x_min - len(object_name) * font_size * 0.6 - 5, y_min),  # Left-middle
				(x_max + 5, y_min),           # Right-middle
				(x_min, y_min + (y_max - y_min) / 2),  # Left-center
				(x_max + 5, y_min + (y_max - y_min) / 2),  # Right-center
			]
			
			# Check each position for overlaps and boundary constraints
			for pos in positions:
				x, y = pos
				
				# Check image boundaries
				if (x < 0 or y < 0 or 
					x + len(object_name) * font_size * 0.6 > img_width or 
					y + font_size * 1.2 > img_height):
					continue
				
				# Check for overlaps with existing labels
				overlap_found = False
				for existing_pos, existing_text in existing_positions:
					if _check_label_overlap(pos, existing_pos, object_name, existing_text, font_size):
						overlap_found = True
						break
				
				if not overlap_found:
					return pos
			
			# If all positions have overlaps, return the default position with slight offset
			offset_y = len(existing_positions) * (font_size * 1.5)
			return (x_min, max(0, y_min - 5 - offset_y))
		
		# Create a copy of the image
		display_image = self.image.copy()
		
		# Create figure and axis
		fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
		ax.imshow(display_image)
		ax.axis('off')
		ax.grid(False)
		
		# Generate distinct colors for each bounding box
		colors = plt.cm.tab20(np.linspace(0, 1, len(bboxes_dict)))
		
		# Keep track of label positions to avoid overlaps
		label_positions = []
		
		# Plot each bounding box
		for i, (object_name, bbox) in enumerate(bboxes_dict.items()):
			x_min, x_max, y_min, y_max = bbox
			
			# Ensure coordinates are valid
			x_min, x_max = max(0, min(x_min, x_max)), min(self.width, max(x_min, x_max))
			y_min, y_max = max(0, min(y_min, y_max)), min(self.height, max(y_min, y_max))
			
			# Calculate width and height
			width = x_max - x_min
			height = y_max - y_min
			
			# Skip invalid bounding boxes
			if width <= 0 or height <= 0:
				continue
				
			# Create rectangle patch
			color = colors[i % len(colors)]
			rect = patches.Rectangle((x_min, y_min), width, height,
								linewidth=line_thickness, edgecolor=color, 
								facecolor='none', alpha=1.0)
			ax.add_patch(rect)
			
			# Add label if requested
			if show_labels:
				# Find best position for label to avoid overlaps
				label_pos = _find_best_label_position(
					bbox, object_name, label_positions, font_size, self.width, self.height
				)
				label_x, label_y = label_pos
				
				# Store this label position
				label_positions.append((label_pos, object_name))
				
				# Create text with background for better visibility
				text = ax.text(label_x, label_y, object_name, 
							fontsize=font_size, color='white', weight='bold',
							bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=alpha))
				
				# Add path effects for better readability
				text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
		
		# Set title
		if not title:
			plt.title(f"Image {self.ID} - Object Detection Results")
		else:
			plt.title(title)
		
		plt.tight_layout()
		plt.show()
		
		return display_image


	def plot_masks_from_objects(self, mask_objects, figsize=(12, 8), dpi=300, title=None, 
                           line_thickness=2, font_size=10, show_labels=True, alpha=0.7,
                           show_scores=False, score_threshold=0.0):

		
		def _check_label_overlap(pos1, pos2, text1, text2, font_size):
			"""Check if two label positions would overlap"""
			# Approximate text dimensions (rough estimation)
			char_width = font_size * 0.6
			char_height = font_size * 1.2
			
			w1, h1 = len(text1) * char_width, char_height
			w2, h2 = len(text2) * char_width, char_height
			
			x1, y1 = pos1
			x2, y2 = pos2
			
			# Check if rectangles overlap
			return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)
		
		def _find_best_label_position(bbox, object_name, existing_positions, font_size, img_width, img_height):
			"""Find the best position for a label to avoid overlaps"""
			x_min, x_max, y_min, y_max = bbox
			
			# Define possible positions around the bounding box
			positions = [
				(x_min, y_min - 5),           # Top-left (default)
				(x_min, y_max + 15),          # Bottom-left
				(x_max - len(object_name) * font_size * 0.6, y_min - 5),  # Top-right
				(x_max - len(object_name) * font_size * 0.6, y_max + 15), # Bottom-right
				(x_min - len(object_name) * font_size * 0.6 - 5, y_min),  # Left-middle
				(x_max + 5, y_min),           # Right-middle
				(x_min, y_min + (y_max - y_min) / 2),  # Left-center
				(x_max + 5, y_min + (y_max - y_min) / 2),  # Right-center
			]
			
			# Check each position for overlaps and boundary constraints
			for pos in positions:
				x, y = pos
				
				# Check image boundaries
				if (x < 0 or y < 0 or 
					x + len(object_name) * font_size * 0.6 > img_width or 
					y + font_size * 1.2 > img_height):
					continue
				
				# Check for overlaps with existing labels
				overlap_found = False
				for existing_pos, existing_text in existing_positions:
					if _check_label_overlap(pos, existing_pos, object_name, existing_text, font_size):
						overlap_found = True
						break
				
				if not overlap_found:
					return pos
			
			# If all positions have overlaps, return the default position with slight offset
			offset_y = len(existing_positions) * (font_size * 1.5)
			return (x_min, max(0, y_min - 5 - offset_y))
		
		# Filter masks by score threshold
		filtered_masks = [mask for mask in mask_objects if mask.score >= score_threshold]
		
		if not filtered_masks:
			print(f"No masks found above score threshold {score_threshold}")
			return self.image.copy()
		
		# Create a copy of the image
		display_image = self.image.copy()
		
		# Create figure and axis
		fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
		ax.imshow(display_image)
		ax.axis('off')
		ax.grid(False)
		
		# Generate distinct colors for each mask
		colors = plt.cm.tab20(np.linspace(0, 1, len(filtered_masks)))
		
		# Keep track of label positions to avoid overlaps
		label_positions = []
		
		# Sort masks by score (highest first) for better visualization
		sorted_masks = sorted(filtered_masks, key=lambda x: x.score, reverse=True)
		
		# Plot each mask
		for i, mask_obj in enumerate(sorted_masks):
			# Get bounding box coordinates
			x_min, x_max = mask_obj.x_min, mask_obj.x_max
			y_min, y_max = mask_obj.y_min, mask_obj.y_max
			
			# Ensure coordinates are valid
			x_min, x_max = max(0, min(x_min, x_max)), min(self.width, max(x_min, x_max))
			y_min, y_max = max(0, min(y_min, y_max)), min(self.height, max(y_min, y_max))
			
			# Calculate width and height
			width = x_max - x_min
			height = y_max - y_min
			
			# Skip invalid bounding boxes
			if width <= 0 or height <= 0:
				continue
			
			# Create rectangle patch
			color = colors[i % len(colors)]
			rect = patches.Rectangle((x_min, y_min), width, height,
								linewidth=line_thickness, edgecolor=color, 
								facecolor='none', alpha=1.0)
			ax.add_patch(rect)
			
			# Add label if requested
			if show_labels:
				# Determine label text
				if mask_obj.most_freq_label:
					label_text = mask_obj.most_freq_label
				elif mask_obj.prompt:
					label_text = mask_obj.prompt
				else:
					label_text = f"Mask_{mask_obj.ID}"
				
				# Add score if requested
				if show_scores:
					label_text += f" ({mask_obj.score:.2f})"
				
				# Find best position for label to avoid overlaps
				bbox = [x_min, x_max, y_min, y_max]
				label_pos = _find_best_label_position(
					bbox, label_text, label_positions, font_size, self.width, self.height
				)
				label_x, label_y = label_pos
				
				# Store this label position
				label_positions.append((label_pos, label_text))
				
				# Create text with background for better visibility
				text = ax.text(label_x, label_y, label_text, 
							fontsize=font_size, color='white', weight='bold',
							bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=alpha))
				
				# Add path effects for better readability
				text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
		
		# Set title
		if not title:
			if len(filtered_masks) != len(mask_objects):
				plt.title(f"Image {self.ID} - Masks (showing {len(filtered_masks)}/{len(mask_objects)} above threshold {score_threshold})")
			else:
				plt.title(f"Image {self.ID} - Masks ({len(filtered_masks)} total)")
		else:
			plt.title(title)
		
		plt.tight_layout()
		plt.show()
		
		return display_image

	
	def get_wordnet_pos(self, word):
		"""Map POS tag to first character lemmatize() accepts"""
		tag = nltk.pos_tag([word])[0][1][0].upper()
		tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
		return tag_dict.get(tag, wordnet.NOUN)


	def disambiguate_overlapping_masks(self, grouped_df : pd.DataFrame, sim_threshold=0.93):

		# Find and handle lemma duplicates
		plurals = []
		labels_to_remove = set()
		processed_pairs = set() 
		
		lemmatizer = WordNetLemmatizer()
		
		# Compute area for comparison
		grouped_df['area'] = (grouped_df['x_max'] - grouped_df['x_min']) * (grouped_df['y_max'] - grouped_df['y_min'])

		for i in range(len(grouped_df)):
			label1_orig = grouped_df.iloc[i]['label']
			
			# Skip if this label is already marked for removal
			if label1_orig in labels_to_remove:
				continue
				
			# Get the lemmatized version
			label1_lemma = lemmatizer.lemmatize(label1_orig, self.get_wordnet_pos(label1_orig))
			
			for j in range(i + 1, len(grouped_df)):  # Start from i+1 to avoid self-comparison and duplicates
				label2_orig = grouped_df.iloc[j]['label']
				
				# Skip if this label is already marked for removal
				if label2_orig in labels_to_remove:
					continue
				
				# Create a pair key to avoid processing the same pair twice
				pair_key = tuple(sorted([label1_orig, label2_orig]))
				if pair_key in processed_pairs:
					continue
				processed_pairs.add(pair_key)
				
				# Get the lemmatized version
				label2_lemma = lemmatizer.lemmatize(label2_orig, self.get_wordnet_pos(label2_orig))
				
				similarity = Levenshtein.jaro_winkler(label1_lemma, label2_lemma)
				
				if (label1_lemma == label2_lemma) or (similarity > sim_threshold):
					
					# Find which label has the smaller area
					area1 = grouped_df.iloc[i]['area']
					area2 = grouped_df.iloc[j]['area']
					
					if area1 > area2:
						label_to_remove = label1_orig
						label_to_keep = label2_orig
					else:
						label_to_remove = label2_orig
						label_to_keep = label1_orig
					
					labels_to_remove.add(label_to_remove)
					plurals.append((label1_orig, label2_orig, f"kept: {label_to_keep}"))
					
					print(f"  -> Lemma match! Removing '{label_to_remove}', keeping '{label_to_keep}'")

		grouped_df = grouped_df[~grouped_df['label'].isin(labels_to_remove)].reset_index(drop=True)
		
		# Drop the area column as it's no longer needed
		grouped_df.drop(columns=['area'], inplace=True)

		# for debugging
		print(f"\nPlural/duplicate pairs found: {plurals}")

		return grouped_df


	def get_scene_masks(self, sim_threshold=0.93):

		columns = ["label", "x_min", "x_max", "y_min", "y_max"]
		df_masks = pd.DataFrame(columns=columns)
		
		# Extract mask data
		for indx in range(len(self.masks)):
			mask = self.masks[indx]
			df_masks.loc[indx] = [mask.most_freq_label, mask.x_min, mask.x_max, mask.y_min, mask.y_max]

		# Order by label alphabetically
		df_masks = df_masks.sort_values(by="label")
		df_masks.reset_index(drop=True, inplace=True)

		# Group by label and take average of coordinates
		grouped_df = df_masks.groupby('label').agg({
			'x_min': 'min', 
			'x_max': 'min', 
			'y_min': 'min', 
			'y_max': 'min'
		}).reset_index()

		# Round coordinates
		for col in ['x_min', 'x_max', 'y_min', 'y_max']:
			grouped_df[col] = grouped_df[col].round(4)
		
		grouped_df = self.disambiguate_overlapping_masks( grouped_df, sim_threshold=sim_threshold )
		
		print(f"Final unique labels: {sorted(grouped_df['label'].unique())}")
		
		# convert to dictionary where the label is the key
		label_dict = grouped_df.set_index('label').T.to_dict('list')

		self.mask_labels = label_dict
		
		return label_dict

	

	