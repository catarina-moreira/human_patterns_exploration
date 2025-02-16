import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os

import math

import matplotlib.patheffects as path_effects



class ImageData:
    def __init__(self, path, distance_cm=60.0, angle_width_deg=38.1, angle_height_deg=28.6, refresh_rate_hz=100, sampling_rate_hz=1000, nominal_monitor_inch=21.0, target_dpi=100):
        self.ID = os.path.basename(path)
        self.path = path
        self.width = None
        self.height = None

        self.image = self.load()

        self.masks = {}

    def load(self, img_type="rgb"):
        """Load and return the image as a numpy array"""

        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"Image not found: {self.path}")

        image = cv2.imread(self.path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not load the image with cv2")

        # Convert image from BGR (OpenCV default) to RGB
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.height, self.width, _ = image.shape

        return image
        
    def plot(self, figsize=(8,10)):
        """Display the image"""
        plt.grid(False)
        plt.axis('off')
        plt.gcf().set_size_inches(figsize[0],figsize[1])
        plt.imshow(self.image)


    def get_best_mask(self):
        """Returns the mask with the highest score for this image."""
        if not self.masks:
            return None  # No masks available

        # Select the mask with the highest score
        best_mask = max(self.masks.values(), key=lambda m: m.score)
        return best_mask


    def draw_fixations(self, fix, alpha=0.5, figsize=(12, 8), dpi=100, savefilename=None, 
                    fix_color="#729fcf", fix_edge_color="#204a87", size = None):

        img = self.image

        x = fix['X']
        y = fix['Y']

        if size is None:
            size = fix['FixationDurationNorm']
        else:
            size = size
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)  
        ax.imshow(img)
        ax.scatter(x, y, s=size, c=fix_color, alpha=alpha, edgecolors=fix_edge_color)
        ax.grid(False)
        ax.axis('off')

        # Save if needed
        if savefilename:
            plt.savefig(savefilename, bbox_inches='tight')
        plt.show()



    def draw_scanpath(self, fix, width=1, alpha=0.5, alpha_font=1, figsize=(12, 8), dpi=100, savefilename=None, 
                    fix_color="#729fcf", fix_edge_color="#204a87", font_color="white", fontsize=10, size=None):
        
        img = self.image

        if "Indx" in fix.columns:
            fix_unique = fix.groupby('Indx').first().reset_index()
            x_unique = fix_unique['X']
            y_unique = fix_unique['Y']

            x = fix['X']
            y = fix['Y']
            
            if size is None:
                size = fix['FixationDurationNorm']
        else:
            x_unique = fix['X']
            y_unique = fix['Y']

            x = fix['X']
            y = fix['Y']

            if size is None:
                size = fix['FixationDurationNorm']

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

        ax.grid(False)
        ax.axis('off')

        if savefilename:
            plt.savefig(savefilename, bbox_inches='tight')
        plt.show()



    def draw_display(self, dispsize=None, dpi=100):
        """
        Loads an image with OpenCV (BGR), converts it to RGB,
        and plots it on a Matplotlib figure.
        """

        img = self.image
        w, h = self.width, self.height

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

        figsize = (dispsize[0]/dpi, dispsize[1]/dpi)
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

    def draw_heatmap(self, fix, alpha=0.5, savefilename=None, title=None, cmap="viridis"):

        # Copy the background image so we don't alter the original
        img_copy = self.image.copy()

        # We'll use the actual image dimensions for display size
        dispsize = (self.width, self.height)

        # Create a figure and axis by calling your image-drawing method
        # (assuming `image.draw_display(dispsize)` returns (fig, ax))
        fig, ax = self.draw_display(dispsize)

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
                        gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * fix['FixationDurationNorm'][i]
                except:
                    pass
            else:
                # Fully in-bounds, just add the Gaussian
                heatmap_fixations[y_pos:y_pos+gwh, x_pos:x_pos+gwh] += \
                    gaus * fix['FixationDurationNorm'][i]

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
