import pandas as pd

def normalize_fixation_duration(df):
    # normalize fixation duration
    durations = df['FixationDuration']
    minval, maxval = durations.min(), durations.max()

    # Normalization: scale durations to 0..100 range:
    df['FixationDurationNorm'] = 100 * (durations - minval) / (maxval - minval + 1e-9)

    return df

def filter_data(df, img_id, condition):

    #print("Data size before filtering:", df.shape)
    filtered_data = df[(df['ItemNum'] == img_id) & (df['Condition'] == condition)]
    gaze_participant = filtered_data[['ParticipantID','X', 'Y', 'FixationDuration']].copy()
    #print("Data size before removing bias:", gaze_participant.shape)

    gaze_participant.reset_index(inplace=True)
    gaze_participant.drop(columns=['index'], inplace=True)

    return gaze_participant

def remove_center_bias(df, seconds=1, rows_per_second=3):
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
    return df.groupby('ParticipantID').apply(lambda x: x.iloc[rows_to_remove:]).reset_index(drop=True)