
import os
import pickle

from classes.ImageData import ImageData
from classes.SAM_Segmentation import SAM_Segmentation

from utils.experiment_data import filter_data, normalize_fixation_duration, remove_center_bias
from utils.prompts import sample_points_circle, sample_points_triangle, sample_points_cross

HOME = os.getcwd()
DATA_DIR = os.path.join(HOME, "data", "experiments")
IMAGE_DIR = os.path.join(HOME, "data", "images")
RESULTS_DIR = os.path.join(HOME, "outputs")

SAM_CONFIG = f"configs/sam2.1/sam2.1_hiera_l.yaml"
SAM_MODEL = f"{HOME}/checkpoints/sam2.1_hiera_large.pt"

def segment_image(img_id, data, condition_id, center_bias_seconds, img_type="unexp", prompt_type="point", prompt_diameter=10, prompt_n_points=5, arm_length_horizontal=11, arm_length_vertical=11):


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
            X = participant_data_filtered["X"].values
            Y = participant_data_filtered["Y"].values

            # if it is a single point, convert to list
            try:
                len(X)
            except:
                X = [X]
                Y = [Y]

            print(X,Y)
            
            mask = sam.compute_masks_with_prompt(X, Y, indx)
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


def save_image_data(img_data, img_id, df, img_type, participant_id, prompt_type="point"):
    # Ensure the directory exists
    dir_path = os.path.join(RESULTS_DIR, "masks_gaze_driven", prompt_type)
    os.makedirs(dir_path, exist_ok=True)

    # Save both image data and DataFrame in a pickle file
    pickle_path = os.path.join(dir_path, f"IMG_{img_id}_{img_type}_Part_{participant_id}.pkl")
    with open(pickle_path, 'wb') as file:
        pickle.dump({"img_data": img_data, "df": df}, file)