import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import imageio.v2 as imageio
from PIL import Image
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os
import pickle
from classes.ImageData import ImageData

def animate_fixations(csv_path, participant_id, output_path):
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Filter data for the given participant_id
    df_participant = df[df['participant_id'] == participant_id].copy()
    
    if df_participant.empty:
        print(f"No data found for participant {participant_id}")
        return
    
    # Ensure X, Y, bbox coordinates are valid (not NaN and numeric)
    df_participant.dropna(subset=['X', 'Y', 'xmin', 'xmax', 'ymin', 'ymax', 'best_label', 'mask_path'], inplace=True)
    df_participant[['X', 'Y', 'xmin', 'xmax', 'ymin', 'ymax']] = df_participant[['X', 'Y', 'xmin', 'xmax', 'ymin', 'ymax']].apply(pd.to_numeric, errors='coerce')
    
    if len(df_participant) < 2:
        print(f"Not enough valid fixation points for animation (found {len(df_participant)}).")
        return
    
    # Load the image and check if it exists
    image_path = df_participant.iloc[0]['image_path']
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return
    
    image = Image.open(image_path).convert("RGBA")
    img_width, img_height = image.size
    
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.grid(False)
    ax.axis('off')
    
    # Initialize elements
    fixation_dot, = ax.plot([], [], 'ro', markersize=6)
    bbox_patch = patches.Rectangle((0, 0), 0, 0, linewidth=2, edgecolor='yellow', facecolor='none')
    ax.add_patch(bbox_patch)
    label_text = ax.text(10, 10, '', fontsize=12, color='white', backgroundcolor='black')
    arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='red', lw=2))
    mask_overlay = ax.imshow(np.zeros((img_height, img_width, 4), dtype=np.uint8), alpha=0)  # Placeholder for mask overlay
    
    def update(frame):
        if frame >= len(df_participant):
            return []  # Avoid out-of-range error
        
        row = df_participant.iloc[frame]
        
        # Ensure numeric values
        try:
            x, y = float(row['X']), float(row['Y'])
            xmin, xmax, ymin, ymax = float(row['xmin']), float(row['xmax']), float(row['ymin']), float(row['ymax'])
        except ValueError:
            print(f"Invalid numeric values in frame {frame}, skipping.")
            return []
        
        # Load and overlay mask from pickle file
        mask_path = row['mask_path'].replace("png", "pkl")
        if os.path.exists(mask_path):
            with open(mask_path, 'rb') as f:
                try:
                    mask_data = pickle.load(f)
                    mask = mask_data.cropped_mask  # Extract mask from pickle object
                    mask = np.array(mask, dtype=np.uint8)  # Ensure it is uint8
                    mask_binary = mask > 0  # Convert to boolean mask
                    
                    # Create an RGBA overlay the same size as the image
                    mask_colored = np.zeros((img_height, img_width, 4), dtype=np.uint8)
                    
                    # Map the mask to its correct bounding box position
                    mask_xmin, mask_xmax = mask_data.x_min, mask_data.x_max
                    mask_ymin, mask_ymax = mask_data.y_min, mask_data.y_max
                    
                    mask_resized = np.zeros((img_height, img_width), dtype=np.uint8)
                    mask_resized[mask_ymin:mask_ymax, mask_xmin:mask_xmax] = mask
                    
                    # Apply blue color to mask
                    mask_colored[mask_resized > 0] = [0, 0, 255, 100]  # Blue with transparency
                    
                    mask_overlay.set_data(mask_colored)
                    mask_overlay.set_alpha(0.8)  # Adjust transparency
                    fig.canvas.draw()  # Ensure the last frame updates properly
                except Exception as e:
                    print(f"Error loading mask file {mask_path}: {e}")
        
        # Update fixation dot
        fixation_dot.set_data([x], [y])  # Ensuring sequence format
        
        # Update bounding box
        bbox_patch.set_xy((xmin, ymin))
        bbox_patch.set_width(xmax - xmin)
        bbox_patch.set_height(ymax - ymin)
        
        # Update label text
        label_text.set_text(row['best_label'])
        label_text.set_position((xmin, ymin - 10))
        
        # Update arrow from previous fixation to the current one
        if frame > 0:
            prev_row = df_participant.iloc[frame - 1]
            prev_x, prev_y = float(prev_row['X']), float(prev_row['Y'])
            arrow.set_position((prev_x, prev_y))
            arrow.xy = (x, y)
        
        return fixation_dot, bbox_patch, label_text, arrow, mask_overlay
    
    ani = FuncAnimation(fig, update, frames=len(df_participant), interval=1000, repeat=False)
    
    # Save animation to MP4 with error handling
    try:
        writer = FFMpegWriter(fps=0.8, metadata={"artist": "Matplotlib"})
        ani.save(output_path, writer=writer)
        print(f"Animation saved to {output_path}")
    except Exception as e:
        print(f"Error saving animation: {e}")
    
    plt.show()
    
    return ani

# Example usage:
csv_file = os.path.join(RESULTS_DIR, "masks_gaze_driven", "best_mask_labeling", "best_labels_gpt-4o_exp.csv")
participant_id = 21
output_video = f"fixation_animation_{participant_id}.mp4"
animate_fixations(csv_file, participant_id, output_video)
