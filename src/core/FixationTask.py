#!/usr/bin/python
# -*- coding: UTF-8 -*-
from src.core.Participant import Participant
from src.core.ImageData import ImageData
import src.utils.style as stl

from src.utils.data_utils import process_participant_data

import matplotlib.patheffects as path_effects

from typing import List

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

class FixationTask:
    
	def __init__(self, participant : Participant, imageData : ImageData, data : pd.DataFrame, 
				condition = None, group = None, seconds = 1, rows_per_second = 3, participant_threshold = 3):
		self.group = group
		self.condition = condition
		self.participant = participant
		self.imageData = imageData
		self.image = self.imageData.image
		self.data = process_participant_data( data, self.imageData.ID, self.participant.ID, 
											condition = condition, group = group, seconds=seconds, 
											rows_per_second=rows_per_second, participant_threshold=participant_threshold)

		self.X = self.data['X']
		self.Y = self.data['Y']
		self.duration = self.data['FixationDuration']
		self.masks = {} # mask generator will update this dictionary
		self.time_on_target = None

	def compute_time_on_target(self):
		"""
		Computes the total time the participant fixated within the target area.
		
		Returns:
		- float: Total fixation duration on target area
		- dict: Dictionary with detailed information including:
			- 'total_time': Total time on target
			- 'fixation_count': Number of fixations on target
			- 'percentage': Percentage of total fixation time spent on target
			- 'target_fixations': DataFrame of fixations that fell within target
		"""
		
		# Check if target coordinates are available
		if self.imageData.target is None:
			raise ValueError("No target coordinates defined for this image")
		
		if len(self.imageData.target) != 4:
			raise ValueError("Target should contain exactly 4 coordinates [x1, x2, y1, y2]")
		
		# Get target coordinates
		x1, x2, y1, y2 = self.imageData.target
		
		# Ensure coordinates are in correct order
		x_min, x_max = min(x1, x2), max(x1, x2)
		y_min, y_max = min(y1, y2), max(y1, y2)
		
		# Filter fixations that fall within the target area
		target_mask = (
			(self.X >= x_min) & 
			(self.X <= x_max) & 
			(self.Y >= y_min) & 
			(self.Y <= y_max)
		)
		
		# Get fixations on target
		target_fixations = self.data[target_mask]
		
		# Calculate total time on target
		time_on_target = target_fixations['FixationDuration'].sum()
		
		# Calculate additional statistics
		total_fixation_time = self.duration.sum()
		fixation_count_on_target = len(target_fixations)
		total_fixation_count = len(self.data)
		percentage_time = (time_on_target / total_fixation_time * 100) if total_fixation_time > 0 else 0
		percentage_count = (fixation_count_on_target / total_fixation_count * 100) if total_fixation_count > 0 else 0
		
		# Store result in instance variable for later access
		self.time_on_target = time_on_target
		
		# Return detailed results
		results = {
			'total_time': time_on_target,
			'fixation_count': fixation_count_on_target,
			'total_fixations': total_fixation_count,
			'percentage_time': percentage_time,
			'percentage_count': percentage_count,
			'target_fixations': target_fixations,
			'target_coordinates': {
				'x_min': x_min, 'x_max': x_max,
				'y_min': y_min, 'y_max': y_max
			}
		}
		
		return results


	def get_participant_ids(self, filtered_data):
		return filtered_data['ParticipantID'].unique()

	def draw_fixations(self, df = None, alpha=0.5, figsize=(12, 8), dpi=300, savefilename=None, fix_color="#729fcf", fix_edge_color="#204a87", size = None, title = None):

		if not fix_color[0] == "#":
			fix_color = stl.COLORS[fix_color][0]
            
		if not fix_edge_color[0] == "#":
			fix_edge_color = stl.COLORS[fix_edge_color][0]

		if df is None:
			fix = self.data 
		else:
			fix = df  

		x = fix.X
		y = fix.Y

		if not size:
			size = self.duration*0.2
		
		fig, ax = plt.subplots(figsize=figsize, dpi=dpi)  
		ax.imshow(self.image)
		ax.scatter(x, y, s=size, c=fix_color, alpha=alpha, edgecolors=fix_edge_color)
		if not title:
			ax.set_title(f"Participant {self.participant.ID} Fixations for Image {self.imageData.ID}")
		else:
			ax.set_title(title)
		ax.grid(False)
		ax.axis('off')

		# Save if needed
		if savefilename:
				plt.savefig(savefilename, bbox_inches='tight')
				plt.show()

	def draw_scanpath(self, df = None, width=1, alpha=0.5, alpha_font=1, figsize=(12, 8), dpi=300, savefilename=None, 
                    fix_color="#729fcf", fix_edge_color="#204a87", font_color="#FFFFFF", fontsize=10, size=None, title = None):
        
        
		if not fix_color[0] == "#":
			fix_color = stl.COLORS[fix_color][0]
            
		if not fix_edge_color[0] == "#":
			fix_edge_color = stl.COLORS[fix_edge_color][0] 
        
		if not font_color[0] == "#":
			font_color = stl.COLORS[font_color][1]
        
		if df is None:
			fix = self.data 
		else:
			fix = df          
        
		img = self.image

		
		if "Indx" in fix.columns:
			fix_unique = fix.groupby('Indx').first().reset_index()
			x_unique = fix_unique['X']
			y_unique = fix_unique['Y']

			x = fix['X']
			y = fix['Y']
            
			if size is None:
				size = fix['FixationDuration']
			
		else:
			x_unique = fix['X']
			y_unique = fix['Y']

			x = fix['X']
			y = fix['Y']

			if size is None:
				size = fix['FixationDuration']

		fig, ax = plt.subplots(figsize=figsize, dpi=dpi)  
		ax.imshow(img)
		ax.scatter(x, y, s=size, facecolors=fix_color, edgecolors=fix_edge_color, alpha=alpha)

		# Draw annotations (fixation numbers)
		for i in range(len(x_unique)):
				ax.annotate(str(i+1), (x_unique.iloc[i], y_unique.iloc[i]-15), color=font_color, alpha=alpha_font, 
                        ha='center', va='center', fontweight='bold', fontsize=fontsize,
                        path_effects=[
                            path_effects.Stroke(linewidth=2, foreground='black'),  # Edge color
                            path_effects.Normal()  # Normal text rendering on top
                        ])

		# Draw arrows
		for i in range(len(x_unique)-1):
				ax.arrow(x_unique.iloc[i], y_unique.iloc[i], x_unique.iloc[i+1] - x_unique.iloc[i], y_unique.iloc[i+1] - y_unique.iloc[i], 
                    alpha=alpha, fc=fix_color, ec=fix_color, fill=True, shape='full',
                    width=width, head_width=8, head_length=8, overhang=0.3)
		if not title:
			ax.set_title(f"Participant {self.participant.ID} Scanpath for Image {self.imageData.ID}")
		else:
			ax.set_title(title)
		ax.grid(False)
		ax.axis('off')

		if savefilename:
				plt.savefig(savefilename, bbox_inches='tight')
		plt.show()

	def draw_display(self, dispsize=None, dpi=300, figsize = (12,8)):

		img = self.image
		w, h = self.imageData.width, self.imageData.height

		# If dispsize not given, use the image size
		if dispsize is None:
				dispsize = (w, h)  # (width, height)

		# Create a screen (3D for color)
		screen = np.zeros((dispsize[1], dispsize[0], 3), dtype=img.dtype)

		# Center coordinates
		y = int(dispsize[1]/2 - h/2)
		x = int(dispsize[0]/2 - w/2)

		# Place the RGB image on the screen
		screen[y:y+h, x:x+w] = img

		#figsize = (dispsize[0]/dpi, dispsize[1]/dpi)
		fig = plt.figure(figsize=figsize, dpi=dpi, frameon=False)
		ax = plt.Axes(fig, [0, 0, 1, 1])
		ax.set_axis_off()
		fig.add_axes(ax)

		# Show
		ax.imshow(screen)
		return fig, ax
        
	def gaussian(self, x, sx, y=None, sy=None):
		# square Gaussian if only x values are passed
		if y == None:
				y = x
		if sy == None:
				sy = sx
		# centers
		xo = x/2
		yo = y/2
		# matrix of zeros
		M = np.zeros([y, x], dtype=float)
		# gaussian matrix
		for i in range(x):
				for j in range(y):
						M[j, i] = np.exp(-1.0 * (((float(i)-xo)**2/(2*sx*sx)) + ((float(j)-yo)**2/(2*sy*sy))))
		return M

	def draw_heatmap(self, df = None, alpha=0.5, savefilename=None, title=None, cmap="viridis", dispsize=None, dpi = 300, figsize=(12,8)):

			if df is None:
				fix = self.data.copy()
			else:
				fix = df  

			# We'll use the actual image dimensions for display size
			if dispsize is None:
				dispsize = (self.imageData.width, self.imageData.height)

			fig, ax = self.draw_display(dispsize = dispsize, dpi = dpi, figsize = figsize)

			# Generate the Gaussian "kernel"
			gwh = 200  # Gaussian window size
			gsdwh = gwh / 6.0
			gaus = self.gaussian(gwh, gsdwh)

			# Prepare a larger heatmap array with some border (strt)
			strt = gwh // 2
			heatmapsize = (dispsize[1] + 2 * strt, dispsize[0] + 2 * strt)
			heatmap_fixations = np.zeros(heatmapsize, dtype=float)

			# Build the heatmap by adding Gaussian distributions at each fixation
			for i in fix.index:
					x_pos = strt + int(fix['X'][i]) - gwh // 2
					y_pos = strt + int(fix['Y'][i]) - gwh // 2

					# Check if the Gaussian window goes out of bounds
					if (not 0 <= x_pos < dispsize[0]) or (not 0 <= y_pos < dispsize[1]):
							# Adjust for boundary
							hadj = [0, gwh]
							vadj = [0, gwh]
							if x_pos < 0:
									hadj[0] = -x_pos
									x_pos = 0
							elif x_pos + gwh > dispsize[0]:
									hadj[1] = gwh - ((x_pos + gwh) - dispsize[0])
							if y_pos < 0:
									vadj[0] = -y_pos
									y_pos = 0
							elif y_pos + gwh > dispsize[1]:
									vadj[1] = gwh - ((y_pos + gwh) - dispsize[1])
							try:
									heatmap_fixations[y_pos:y_pos+vadj[1], x_pos:x_pos+hadj[1]] += \
					gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * self.duration[i]
							except:
									pass
					else:
							# Fully in-bounds, just add the Gaussian
							heatmap_fixations[y_pos:y_pos+gwh, x_pos:x_pos+gwh] += gaus * self.duration[i]

			# Crop the extra border
			heatmap_fixations = heatmap_fixations[strt:dispsize[1]+strt, strt:dispsize[0]+strt]

			# Optionally remove low values below average
			nonzero_vals = heatmap_fixations[heatmap_fixations > 0]
			if len(nonzero_vals) > 0:
					lowbound = np.mean(nonzero_vals)
					heatmap_fixations[heatmap_fixations < lowbound] = np.nan

			# Draw the heatmap on top of the image
			cax = ax.imshow(heatmap_fixations, cmap=cmap, alpha=alpha)

			# OPTIONAL: Add a colorbar to interpret intensity
			# The fraction/pad arguments help position a smaller bar nicely
			fig.colorbar(cax, ax=ax, fraction=0.03, pad=0.04)

			# Title
			ax.set_title(title or "", fontsize=12)

			# Save the figure if a filename is given
			if savefilename:
					fig.savefig(savefilename, bbox_inches='tight')

			return fig, heatmap_fixations

