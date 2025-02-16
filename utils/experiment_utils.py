
import os
import pickle

from PIL import Image

from classes.ImageData import ImageData
from classes.SAM_Segmentation import SAM_Segmentation

from utils.experiment_data import filter_data, normalize_fixation_duration, remove_center_bias
from utils.prompts import sample_points_circle, sample_points_triangle, sample_points_cross

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

HOME = os.getcwd()
DATA_DIR = os.path.join(HOME, "data", "experiments")
IMAGE_DIR = os.path.join(HOME, "data", "images")
RESULTS_DIR = os.path.join(HOME, "outputs")

SAM_CONFIG = os.path.join("configs", "sam2.1", "sam2.1_hiera_l.yaml")
SAM_MODEL =  os.path.join(HOME, "checkpoints", "sam2.1_hiera_large.pt")


def segment_image(img_id, data, condition_id, center_bias_seconds, img_type="unexp", prompt_type="point", prompt_diameter=10, prompt_n_points=5, arm_length_horizontal=11, arm_length_vertical=11, size_threshold = 100):

    img_path = os.path.join(IMAGE_DIR, f"{img_id}{img_type}.jpg")  # Fix file naming issue
    exp_img = ImageData(img_path)

    print("Processing image: ", img_path)

    participant_data = filter_data(data, img_id, condition_id)
    participant_data = normalize_fixation_duration(participant_data)
    participant_data = remove_center_bias(participant_data, seconds=center_bias_seconds)

    participant_ids = participant_data['ParticipantID'].unique()

    for part_id in participant_ids:
        print("\t Processing PARTICIPANT ID: ", part_id)
        
        sam = SAM_Segmentation(SAM_MODEL, SAM_CONFIG, exp_img, part_id)
        print("\t\tDEVICES: ", sam.device)

        participant_data_filtered = participant_data[participant_data['ParticipantID'] == part_id].reset_index(drop=True)

        if prompt_type == "point":
            participant_data_filtered['Indx'] = participant_data_filtered.index
        elif prompt_type == "circle":
            participant_data_filtered = sample_points_circle(participant_data_filtered, prompt_n_points, diameter=prompt_diameter)
        elif prompt_type == "triangle":
            participant_data_filtered = sample_points_triangle(participant_data_filtered, arm_length_horizontal=arm_length_horizontal, arm_length_vertical=arm_length_vertical)
        elif prompt_type == "cross":
            participant_data_filtered = sample_points_cross(participant_data_filtered, arm_length_horizontal=arm_length_horizontal, arm_length_vertical=arm_length_vertical)

        masks = []
        exp_img.masks[part_id] = []
        for indx in participant_data_filtered['Indx'].unique():
            
            print("\t\tComputing mask for gaze index ", indx)
            X = participant_data_filtered[ participant_data_filtered['Indx'] == indx ]['X'].values
            Y = participant_data_filtered[ participant_data_filtered['Indx'] == indx ]['Y'].values

            # if it is a single point, convert to list
            try:
                len(X)
            except:
                X = [X]
                Y = [Y]
            
            mask = sam.compute_masks_with_prompt(X, Y, indx, size_threshold = size_threshold)
            exp_img.masks[part_id][indx] = mask
             
        #flatten the masks
        #exp_img.masks[part_id] = [item for sublist in exp_img.masks[part_id] for item in sublist]
        save_image_data(exp_img, img_id, participant_data_filtered, img_type, part_id, prompt_type=prompt_type)

    return exp_img


def load_image_data(img_id, img_type, participant_id, prompt_type="point"):

    # Construct the file path
    pickle_path = os.path.join(RESULTS_DIR, "masks_gaze_driven", prompt_type, f"IMG_{img_id}_{img_type}_Part_{participant_id}.pkl")

    # Check if the file exists
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

    # Load the data from the pickle file
    with open(pickle_path, 'rb') as file:
        data = pickle.load(file)

    # Extract image data and DataFrame
    img_data = data.get("img_data", None)
    df = data.get("df", None)

    return img_data, df

def load_best_image_data(img_id, img_type, participant_id):

    # list all files in the directory
    files = os.listdir(os.path.join(RESULTS_DIR, "masks_gaze_driven", "best"))
    # filter out the files that are not pickle files
    participant_files = [file for file in files if file.split("_")[4] == participant_id]

    # get only the pickle files
    best_masks = [file for file in participant_files if file.endswith(".pkl")][0]

    results = {}
    for file in best_masks:

        mask_id = file.split("_")[6]
        prompt_type = file.split("_")[7].replace(".pkl", "")
        with open(os.path.join(RESULTS_DIR, "masks_gaze_driven", "best", file), 'rb') as file:
            data = pickle.load(file)
        
        results[mask_id]['mask_data'] = data
        results[mask_id]['mask_path'] = file
        results[mask_id]['mask_image_path'] = file.replace(".pkl", ".png")
        results[mask_id]['prompt_type'] = prompt_type

    # Load the data from the pickle file
    with open(pickle_path, 'rb') as file:
        data = pickle.load(file)

    # Extract image data and DataFrame
    img_data = data.get("img_data", None)
    df = data.get("df", None)

    return img_data, df


def save_image_data(img_data, img_id, df, img_type, participant_id, prompt_type="point"):
    # Ensure the directory exists
    dir_path = os.path.join(RESULTS_DIR, "masks_gaze_driven", prompt_type)
    os.makedirs(dir_path, exist_ok=True)

    # Save both image data and DataFrame in a pickle file
    pickle_path = os.path.join(dir_path, f"IMG_{img_id}_{img_type}_Part_{participant_id}.pkl")
    with open(pickle_path, 'wb') as file:
        pickle.dump({"img_data": img_data, "df": df}, file)

def get_participant_ids( path, img_type ):

    final = []
    # list all files in the directory
    files = os.listdir(path)
    # filter out the files that are not pickle files
    pickle_files = [file for file in files if file.endswith(".pkl")]

    for file in pickle_files:
         if file.split("_")[2] ==  img_type:
              final.append(file)

    # extract the participant ids from the file names
    participant_ids = [file.split("_")[4].replace(".pkl", "") for file in final]
    participant_ids = [int(p_id) for p_id in participant_ids]
    participant_ids = sorted(participant_ids)

    return participant_ids


def load_image_data_types(img_id, participant_id, prompt_types, img_type):

    res = {}
    for prompt_type in prompt_types:
        img_data, _ = load_image_data(img_id, img_type, participant_id=participant_id, prompt_type=prompt_type)
        res[prompt_type] = img_data.masks[participant_id]
    
    return res

def save_as_png(mask, filename="mask.png", save_dir=""):
    """Saves the mask as a transparent PNG."""
    # Convert the mask to a NumPy array
    # Convert NumPy array to a PIL Image
    img = Image.fromarray(mask.cropped_image_with_alpha)

    # Save the image as PNG (which supports transparency)
    img.save(os.path.join(save_dir, filename), format="PNG", dpi=(300,300))
    print(f"Mask saved as {os.path.join(save_dir, filename)}")

def save_mask(mask, filename="mask.png", save_dir=""):
    save_as_png(mask, filename=filename, save_dir=save_dir)
    with open(os.path.join(save_dir, filename), "wb") as f:
        pickle.dump(mask, f)

def find_best_mask(img_id, participant_ids, prompt_types, img_type, area_threshold=30000):

    for participant_id in participant_ids:
        print(participant_id)
        res = load_image_data_types(img_id, participant_id, prompt_types, img_type)

        prompts = list(res.keys())
        masks = res[prompts[0]]

        for mask in range(len(masks)):
            
            best_score = 0
            best_mask = None
            best_prompt_type = None
            best_area = 0
            
            for prompt_type in prompts:
                if res[prompt_type][mask].score > best_score and res[prompt_type][mask].area < area_threshold:
                    best_score = res[prompt_type][mask].score
                    best_mask = res[prompt_type][mask]
                    best_prompt_type = prompt_type
                    best_area = res[prompt_type][mask].area

            # if mask is None, try again
            if best_mask is None:
                best_score = 0
                best_mask = None
                best_prompt_type = None
                best_area = 0
                
                for prompt_type in prompts:
                    if res[prompt_type][mask].score > best_score:
                        best_score = res[prompt_type][mask].score
                        best_mask = res[prompt_type][mask]
                        best_prompt_type = prompt_type
                        best_area = res[prompt_type][mask].area

            best_mask.plot(figsize=(2,2), title=f"{best_prompt_type}: {best_score:.4f} - {best_area:.4f}")
            save_as_png(best_mask, filename=f"IMG_{img_id}_{img_type}_Part_{participant_id}_MASK_{mask}_{best_prompt_type}.png", save_dir=os.path.join(RESULTS_DIR, "masks_gaze_driven", "best"))
            save_mask(best_mask, os.path.join(RESULTS_DIR, "masks_gaze_driven", "best", f"IMG_{img_id}_{img_type}_Part_{participant_id}_MASK_{mask}_{best_prompt_type}_{best_score:.4f}.pkl"))


def get_mask_candidates(img_id, participant_id, prompt_types, img_type):

    res = load_image_data_types(img_id, participant_id, prompt_types, img_type)

    prompts = list(res.keys())
    masks = res[prompts[0]]

    for mask in range(len(masks)):

        print(f"Processing mask {mask} for participant {participant_id}")
        
        curr_score = 0
        curr_mask = None
        curr_prompt_type = None
        curr_area = None

        for prompt_type in prompts:
        
            curr_score = res[prompt_type][mask].score
            curr_mask = res[prompt_type][mask]
            curr_prompt_type = prompt_type
            curr_area = res[prompt_type][mask].area
            curr_mask.plot(figsize=(2,2), title=f"{curr_prompt_type}: {curr_score:.4f} - {curr_area:.4f}")
            plt.show()

def plot_participants_masks(img_data, participant_ids, P_INDX):
    print("PARTICIPANT: ", participant_ids[P_INDX])
    for m_indx in range(len(img_data.masks[participant_ids[P_INDX]])):
        img_data.masks[participant_ids[P_INDX]][m_indx].plot(figsize=(2,2)) 
        plt.show()