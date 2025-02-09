import numpy as np
import pandas as pd


def sample_points_circle(df, N, diameter = 10):
    sampled_points = []

    for index, row in df.iterrows():
        x_center = row['X']
        y_center = row['Y']
        radius = diameter / 2 #(row['FixationDuration'] / downsample_factor)
        participantID = row['ParticipantID']
        fixationDur = row['FixationDuration']
        fixationDurNorm = row['FixationDurationNorm']

        sampled_points.append((participantID, x_center, y_center, fixationDur, fixationDurNorm, index))

        # Generate N points at regular intervals around the circle
        for i in range(N):
            angle = 2 * np.pi * i / N  # Regular interval
            x = x_center + radius * np.cos(angle)
            y = y_center + radius * np.sin(angle)
            sampled_points.append((participantID, x, y, fixationDur, fixationDurNorm, index))


    return pd.DataFrame(sampled_points, columns=['ParticipantID', 'X', 'Y', 'FixationDuration', 'FixationDurationNorm', 'Indx'])

def sample_points_triangle(df, arm_length_horizontal=11, arm_length_vertical = 11):
    sampled_points = []

    for index, row in df.iterrows():
        x_center = row['X']
        y_center = row['Y']

        participantID = row['ParticipantID']
        fixationDur = row['FixationDuration']
        fixationDurNorm = row['FixationDurationNorm']

        # Points at the tips of the cross
        # Horizontal arm tips
        x_right = x_center + arm_length_horizontal
        x_left = x_center - arm_length_horizontal

        # Vertical arm tips
        y_bottom= y_center + arm_length_vertical
        y_top = y_center - arm_length_vertical

        # fixation point
        sampled_points.append((participantID, x_center, y_center, fixationDur, fixationDurNorm, index))

        # right arm tip
        sampled_points.append((participantID, x_right, y_center, fixationDur, fixationDurNorm, index))

        # left arm tip
        sampled_points.append((participantID, x_left, y_center, fixationDur, fixationDurNorm, index))

        # top arm tip
        sampled_points.append((participantID, x_center, y_top, fixationDur, fixationDurNorm, index))

    return pd.DataFrame(sampled_points, columns=['ParticipantID', 'X', 'Y', 'FixationDuration','FixationDurationNorm', 'Indx'])


def sample_points_cross(df, arm_length_horizontal=11, arm_length_vertical = 11):
    sampled_points = []

    for index, row in df.iterrows():
        x_center = row['X']
        y_center = row['Y']

        participantID = row['ParticipantID']
        fixationDur = row['FixationDuration']
        fixationDurNorm = row['FixationDurationNorm']

        # Points at the tips of the cross
        # Horizontal arm tips
        x_right = x_center + arm_length_horizontal
        x_left = x_center - arm_length_horizontal

        # Vertical arm tips
        y_bottom= y_center + arm_length_vertical
        y_top = y_center - arm_length_vertical

        # fixation point
        sampled_points.append((participantID, x_center, y_center, fixationDur, fixationDurNorm, index))

        # right arm tip
        sampled_points.append((participantID, x_right, y_center, fixationDur, fixationDurNorm, index))

        # left arm tip
        sampled_points.append((participantID, x_left, y_center, fixationDur, fixationDurNorm, index))

        # top arm tip
        sampled_points.append((participantID, x_center, y_top, fixationDur, fixationDurNorm, index))

        # bottom arm tip
        sampled_points.append((participantID, x_center, y_bottom, fixationDur, fixationDurNorm, index))

    return pd.DataFrame(sampled_points, columns=['ParticipantID', 'X', 'Y', 'FixationDuration', 'FixationDurationNorm', 'Indx'])
