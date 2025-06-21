import os
import requests


HOME = os.getcwd()
DATA_DIR = os.path.join(HOME, "data", "experiments")
IMAGE_DIR = os.path.join(HOME, "data", "images")
RESULTS_DIR = os.path.join(HOME, "outputs")

SAM_CONFIG = os.path.join("configs", "sam2.1", "sam2.1_hiera_l.yaml")
SAM_MODEL =  os.path.join(HOME, "checkpoints", "sam2.1_hiera_large.pt")


def download_model_weight( filename ):
    # Ensure the directory exists
    os.makedirs('./checkpoints/', exist_ok=True)
    url = 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/' + filename
    filename = './checkpoints/' + filename

    if os.path.exists(filename):
        print(f"File {filename} already exists")
        return

    # Download the file
    response = requests.get(url, stream=True)
    response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded {url} to {filename}")


def download_model_config( filename ):
    # Ensure the directory exists
    os.makedirs('./checkpoints/', exist_ok=True)
    url = 'https://github.com/facebookresearch/sam2/blob/main/sam2/configs/sam2.1/' + filename
    filename = './configs/' + filename

    if os.path.exists(filename):
        print(f"File {filename} already exists")
        return

    # Download the file
    response = requests.get(url, stream=True)
    response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded {url} to {filename}")