import json
from bisect import insort
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from matplotlib.path import Path
from matplotlib.patches import Polygon


# The functions below are all used in sync_events.py

def get_person_info(frame, person_id, key):
    """
    Retrieve specific information (key) for a person based on their person_id.
    
    :param frame: The frame containing the 'Persons' list.
    :param person_id: The ID of the person whose information is being retrieved.
    :param key: The key in the person's dictionary whose value is to be returned (e.g., 'Position', 'TeamSide').
    
    :return: The value of the specified key, or None if not found.
    """
    for person in frame['Persons']:
        if person['Id'] == person_id:
            return person[key]
    return None

def calculate_distance(start_pos, end_pos):
    """
    Calculate the Euclidean distance between the passer and the receiver.
    
    passer_position: (x, z) tuple of passer's position.
    receiver_position: (x, z) tuple of receiver's position.
    
    Returns the distance.
    """
    x1, z1 = start_pos[0], start_pos[2]
    x2, z2 = end_pos[0], end_pos[2]
    return math.sqrt((x2 - x1)**2 + (z2 - z1)**2)

def polar(pos, goal):
    """
    Calculates the distance and angle to the goal from a given position.

    Returns angle (float): Angle to the goal in degrees (relative to the x-axis).
    """
    # Compute angle using atan2 (returns in radians, convert to degrees)
    pos_x, pos_z = pos[0], pos[2]
    goal_x, goal_z = goal[0], goal[2]
    angle = math.degrees(math.atan2(goal_z - pos_z, goal_x - pos_x))
    return angle

def calc_time_ai(clock):
    # Start of the period (1200 seconds for each period)
    period_seconds = (clock['Period'] - 1) * 1200  # Offset for previous periods
    # Calculate time remaining in the current period based on the countdown
    time_remaining_in_period = 1200 - (clock['Minute'] * 60 + clock['Second'] + clock['InjuryTime'])
    # Total time is the time remaining in the current period, adjusted by the period offset
    total_seconds = period_seconds + time_remaining_in_period
    return total_seconds

def actiontype_onehot_from_name(event_name: str, known_actiontypes=None):
    if known_actiontypes is None:
        known_actiontypes = ['EventPass', 'EventShot', 'EventTurnover']  # Add more if needed

    onehot = {}
    for action in known_actiontypes:
        onehot[f'actiontype_{action.lower()}'] = int(event_name == action)
    return onehot

def actionresult_onehot(result):
    return {
        'actionresult_success': int(result == 1),
        'actionresult_miss': int(result == 0)
    }
    
def filter_defenders_in_front(x_a, defenders):
    return [d for d in defenders if d[0] > x_a]
    
def init_opp(frame, teamside):
    opp_pos = []
    for player in frame['Persons']:
        if player['TeamSide'] != teamside:
            x, z = player['Position'][0], player['Position'][2]
            opp_pos.append((x, z))
    return opp_pos

def gaussian_samples(center, sigma, n_samples):
    return np.random.normal(loc=center, scale=sigma, size=(n_samples, 2)).tolist()

def in_triangle(points, triangle_vertices):
    path = Path(triangle_vertices)
    return path.contains_points(points)

def gaussian_mass_in_triangle(center, sigma, triangle, n_samples=10000):
    samples = gaussian_samples(center, sigma, n_samples)
    inside = in_triangle(samples, triangle)
    return sum(inside) / n_samples

def opponent_density(defenders, shot_cone, sigma=0.75):
    total_mass = 0
    for i, d in enumerate(defenders):
        mass = gaussian_mass_in_triangle(d, sigma, shot_cone)
        total_mass += mass
    return total_mass

    
# The functions below are all used in actions_to_gamestates.ipynb

def add_previous_actions_features(df, n=1, exclude_columns=None):
    """
    Appends features from previous n actions to each row in the dataframe.

    Parameters:
    - df: pd.DataFrame of actions
    - n: number of previous actions to include
    - exclude_columns: list of column names to exclude from copying

    Returns:
    - pd.DataFrame with additional columns for previous actions
    """
    df = df.reset_index(drop=True)  # Ensure a clean index
    output = df.copy()
    
    if exclude_columns is None:
        exclude_columns = []

    for i in range(1, n + 1):
        shifted = df.shift(i).add_suffix(f"_prev{i}")
        # Drop excluded columns
        for col in exclude_columns:
            if col in df.columns:
                shifted.drop(f"{col}_prev{i}", axis=1, inplace=True, errors='ignore')
        output = pd.concat([output, shifted], axis=1)

    return output

# The functions below are all used in evaluation.cat

def plot_action_with_history(X, index, history=3, image_path="../hockey_rink.png", ax=None):
    import matplotlib.image as mpimg

    # Load the pitch image
    img = mpimg.imread(image_path)

    # Team color map (0 = green, 1 = purple)
    team_colors = {0: 'green', 1: 'purple'}

    # If no axis is provided, create one
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 7))

    ax.set_title("Action Trajectory")
    ax.imshow(img, extent=[-35.5, 35.5, -18, 18], aspect='auto')

    # Indices of actions: current + history steps before
    indices_to_plot = list(range(max(0, index - history), index + 1))
    alphas = [0.3 + 0.7 * (i / (len(indices_to_plot) - 1)) for i in range(len(indices_to_plot))]

    for i, (row_idx, alpha) in enumerate(zip(indices_to_plot, alphas)):
        row = X.loc[row_idx]
        sx, sz = row["start_x_ai"], row["start_z_ai"]
        ex, ez = row["end_x_ai"], row["end_z_ai"]
        team = row["team"]
        time = row["time"]

        # Determine action type
        if row.get("actiontype_eventpass", 0) == 1:
            action_type = "pass"
        elif row.get("actiontype_eventshot", 0) == 1:
            action_type = "shot"
        elif row.get("actiontype_eventturnover", 0) == 1:
            action_type = "turnover"
        else:
            action_type = "unknown"

        label_text = f"{action_type} ({int(time)})"
        color = team_colors.get(team, 'gray')
        label = "current" if row_idx == index else f"step-{index - row_idx}"

        print(f"{label}: {label_text} | start=({sx}, {sz}), end=({ex}, {ez}), dx={ex - sx}, dz={ez - sz}")

        ax.arrow(sx, sz, ex - sx, ez - sz,
                 color=color, alpha=alpha, head_width=0.5, length_includes_head=True)
        ax.scatter(sx, sz, color=color, alpha=alpha)
        ax.text(sx + 0.5, sz + 0.5, label_text, fontsize=9, color=color, alpha=alpha)

    # Pitch visuals
    ax.set_xlim(-35, 35)
    ax.set_ylim(-15, 15)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
