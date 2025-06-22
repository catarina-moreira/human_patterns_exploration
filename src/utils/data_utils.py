import pandas as pd
import numpy as np


def load_data(data_path : str):
		data = pd.read_csv(data_path)
		return data

def normalize_fixation_duration(df):
    # normalize fixation duration
    durations = df['FixationDuration']
    minval, maxval = durations.min(), durations.max()

    # Normalization: scale durations to 0..100 range:
    df['FixationDurationNorm'] = 100 * (durations - minval) / (maxval - minval + 1e-9)

    return df

def filter_data(data, imgID, partID, condition = None, group = None):
		# Filter by Image first
		filtered = data[data['ItemNum'] == imgID]

		# Filter by Condition and/or Group if present
		if 'Condition' in data.columns:
			filtered = filtered[filtered['Condition'] == condition]
		if 'Group' in data.columns:
			filtered = filtered[filtered['Group'] == group]

		# Filter by ParticipantID unless "ALL" is specified
		if "ALL" not in str(partID):
			filtered = filtered[filtered['ParticipantID'] == partID]

		gaze_participant = filtered_data[['ParticipantID','X', 'Y', 'FixationDuration']].copy()
		gaze_participant.reset_index(inplace=True)
		gaze_participant.drop(columns=['index'], inplace=True)

		return gaze_participant


def process_participant_data(data, imgID, partID, condition = None, group = None, seconds = 1, rows_per_second = 3, participant_threshold = 3):

    filtered_data = filter_data(data, imgID, partID, condition = condition, group = group)
    filtered_data = normalize_fixation_duration(filtered_data)
    filtered_data = remove_center_bias(filtered_data, seconds = seconds, rows_per_second = rows_per_second, participant_threshold = participant_threshold)

    return gaze_participant

def remove_center_bias(df, seconds=1, rows_per_second=3, participant_threshold = 3):
    """
    Removes the first few seconds of data for each participant in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing eye-tracking data.
        seconds (int): Number of seconds to remove from the start for each participant.
        rows_per_second (int): Number of rows that approximate one second of data.

    Returns:
        pd.DataFrame: A DataFrame with the initial seconds of data removed for each participant.
    """
    # Calculate total rows to remove for each participant based on seconds and rows_per_second
    rows_to_remove = round(seconds * rows_per_second)

    # Group the DataFrame by 'ParticipantID' and apply a lambda function to drop the first few rows
    filtered_data = df.groupby('ParticipantID').apply(lambda x: x.iloc[rows_to_remove:]).reset_index(drop=True)

    # only keep the participants that have more than three fixations
    filtered_data = filtered_data[filtered_data.groupby('ParticipantID')['FixationDuration'].transform('count') >= participant_threshold]