'''
The loop at the end of the file consists of 3 steps: sync the game, building the dataframe and writing the dataframe to new file. 

The syncing step has been commented out, because this can be done once. Adding events should be done in this step.
In the second step the features are read/calculated and added to the dataframe.
In final step, the dataframe is written to a file (all_actions_4.0.csv). Note the mode="a", so if you make changes, 
    you'll need to delete the all_actions_4.0.csv first before running this file. 
'''

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

from utils import *

start=time.perf_counter()

def sync_one_game(json_file_path, game_num, file_name):
    # Define a list of relevant event types to be processed
    relevant_events = ['EventPass', 'EventShot', 'EventPossessionChangeTeam', 
                       'EventPossessionChangePlayer', 'EventTurnover', 
                       'EventFaceoff', 'EventGoal']
    
    # Initialize lists and dictionaries for storing timestamps and events
    frame_timestamps = []  # To hold the timestamps of each frame
    events = []            # To hold filtered game events

    # Read the input JSON file, line by line
    with open(json_file_path, "r") as f:

        for line in f:
            # Parse the JSON line into a Python dictionary
            frame = json.loads(line)
            
            # Store the timestamp of the current frame
            frame_timestamps.append(frame['TimestampUTC'])
            
            # Check for the presence of game events
            if 'GameEventsContext' in frame and frame['GameEventsContext']['IdfGameEvents'] != []:
                tmp_list = []
                for i in range(len(frame['GameEventsContext']['IdfGameEvents'])):
                    # Only proceed if the event is one of the relevant types
                    if frame['GameEventsContext']['IdfGameEvents'][i]['Name'] in relevant_events:
                        if (frame['GameEventsContext']['IdfGameEvents'][i]['Name'], frame['GameEventsContext']['IdfGameEvents'][i]['TimestampUTC']) not in tmp_list:
                            tmp_list.append((frame['GameEventsContext']['IdfGameEvents'][i]['Name'], frame['GameEventsContext']['IdfGameEvents'][i]['TimestampUTC']))
                        else:
                            continue
                        # Skip shots that don't involve at least 2 persons
                        if frame['GameEventsContext']['IdfGameEvents'][i]['Name'] == 'EventShot' and len(frame['GameEventsContext']['IdfGameEvents'][i]['Persons']) < 2:
                            continue
                        else:
                            # Append non-shot events directly to events list
                            events.append(frame['GameEventsContext']['IdfGameEvents'][i])

    # Prepare a list of event timestamps for processing
    event_timestamps = []
    for event in events:
        ts = event["TimestampUTC"]
        e = event
        event_timestamps.append((e, ts))

    event_to_frame = {}  # To map event IDs to their corresponding frames

    event_idx = 0  # Index to track current event being processed
    current_event, current_event_ts = event_timestamps[event_idx]
    event_clock = {}  # To track game clocks associated with events
    last_known_clock = {'Period': 1, 'Minute': 20, 'Second': 0, 'InjuryTime': 0, 'IsClockRunning': False}  # Holds the last known clock state
    prev_frame = None  # To hold the previous frame for delta time calculation

    # Collect all unique event IDs for faster lookup
    event_ids = set(event['EventId'] for event in events)

    # Read the input JSON file again to synchronize events to frames
    with open(json_file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            frame_ts = data["TimestampUTC"]  # Get the timestamp of the current frame

            if 'GameClockContext' in data and data['GameClockContext']:
                # Update the last known game clock state
                last_known_clock = data['GameClockContext']  
                
            if 'GameEventsContext' in data and data['GameEventsContext']['IdfGameEvents'] != []:
                for i in range(len(data['GameEventsContext']['IdfGameEvents'])):
                    if data['GameEventsContext']['IdfGameEvents'][i]['Name'] in relevant_events:
                        event_id = data['GameEventsContext']['IdfGameEvents'][i]['EventId']
                        if event_id in event_ids:
                            # If the event ID is recognized, capture the associated game clock
                            if event_id not in event_clock:
                                if 'GameClockContext' in data and data['GameClockContext']:
                                    event_clock[event_id] = data['GameClockContext']
                                else:
                                    event_clock[event_id] = last_known_clock  # Use last known clock if none found

            # Synchronize events with frames based on timestamp
            while event_idx < len(event_timestamps) and frame_ts >= current_event_ts:
                # Calculate the closest frame for the current event based on timestamps
                if prev_frame is not None:
                    delta_prev = abs(prev_frame["TimestampUTC"] - current_event_ts)
                    delta_curr = abs(frame_ts - current_event_ts)
                    best_frame = prev_frame if delta_prev <= delta_curr else data
                    best_delta = min(delta_prev, delta_curr)  # Capture the smallest delta time
                else:
                    best_frame = data
                    best_delta = abs(frame_ts - current_event_ts)  # If first frame, just use this one

                if 'GameClockContext' in best_frame and best_frame['GameClockContext']:
                    original_clock = best_frame['GameClockContext']
                else:
                    original_clock = event_clock.get(current_event['EventId'], last_known_clock)

                # Map the current event ID to the best corresponding frame and clock info
                event_to_frame[current_event['EventId']] = (best_frame, current_event, original_clock)

                # Move to the next event
                event_idx += 1
                if event_idx < len(event_timestamps):
                    current_event, current_event_ts = event_timestamps[event_idx]

                # Store the current frame for the next iteration
                prev_frame = data

    # Output the results into a JSON file formatted with indentation for readability
    with open(f'../processed_data/{file_name}.json', 'w') as outfile:
        json.dump(event_to_frame, outfile, indent=4)

def extract_features(frame, event, clock, home_possession, shot_result, next_frame, next_event, score, game_id, home_strength, visitor_strength):
    features = {}

    # Initialize values in case missing values
    goal_x, goal_y, goal_z = 26.95, 0, 0

    goal_loc = [goal_x, goal_y, goal_z]

    features['game_id'] = game_id
    
    features['timestamp'] = frame['TimestampUTC']

    features['period'] = clock['Period']

    # Action type
    features['actiontype'] = event['Name']

    # One-hot encode action type and update features
    actiontype_features = actiontype_onehot_from_name(event.get('Name'))
    features.update(actiontype_features)

    # One-hot encode result type and update features
    if event['Name'] == 'EventPass' or event['Name'] == 'EventTurnover':
        # Data contains only succesful passes
        result = actionresult_onehot(1)
    elif event['Name'] == 'EventShot':
        result = actionresult_onehot(shot_result)
    features.update(result)

    # Time (GameClockContext) in seconds
    clock_context = calc_time_ai(clock)
    features['time'] = clock_context

    # Start- and endlocation of action
    if event['Name'] == 'EventPass':
        passer_id = event['Persons'][1]['PersonId']
        receiver_id = event['Persons'][2]['PersonId']
        passer_pos =  get_person_info(frame, passer_id, 'Position')
        receiver_pos = get_person_info(frame, receiver_id, 'Position')
        x_start, z_start = passer_pos[0],  passer_pos[2]
        x_end, z_end = receiver_pos[0], receiver_pos[2]
    elif event['Name'] == 'EventShot':
        for person in event['Persons']:
            if person['Role'] == 'MapRoleShooter':
                shooter_id = person['PersonId']
                break  # Stop searching once we find the shooter
        shooter_pos = get_person_info(frame, shooter_id, 'Position')
        x_start, z_start = shooter_pos[0], shooter_pos[2]
        x_end, z_end = shooter_pos[0], shooter_pos[2]
    elif event['Name'] == 'EventTurnover':
        for person in event['Persons']:
            if person['Role'] == 'MapRoleCausedTurnover':
                winner_id = person['PersonId']
                break  # Stop searching once we find the winner
        winner_pos = get_person_info(frame, winner_id, 'Position')
        x_start, z_start = winner_pos[0], winner_pos[2]
        x_end, z_end = winner_pos[0], winner_pos[2]
        
    # Team possesion in current action
    if event['Name'] == 'EventPass':
        side = get_person_info(frame, passer_id, 'TeamSide')
        if side == 1:
            home_possession = True
        else:
            home_possession = False
    elif event['Name'] == 'EventShot':
        side = get_person_info(frame, shooter_id, 'TeamSide')
        if side == 1:
            home_possession = True
        else:
            home_possession = False
    elif event['Name'] == 'EventTurnover':
        side = get_person_info(frame, winner_id, 'TeamSide')
        if side == 1:
            home_possession = True
        else:
            home_possession = False
    features['team'] = home_possession
    
    if event['Name'] == 'EventShot':
        if x_start < 0:  # Shooter is on the right half
             goal_loc = [-26.95, 0, 0]

    # Start location of next action
    if next_event['Name'] == 'EventPass':
        next_passer_id = next_event['Persons'][1]['PersonId']
        next_passer_pos =  get_person_info(next_frame, next_passer_id, 'Position')
        next_x_start, next_z_start = next_passer_pos[0],  next_passer_pos[2]
    elif next_event['Name'] == 'EventShot':
        for person in next_event['Persons']:
            if person['Role'] == 'MapRoleShooter':
                next_shooter_id = person['PersonId']
                break  # Stop searching once we find the shooter
        next_shooter_pos = get_person_info(next_frame, next_shooter_id, 'Position')
        next_x_start, next_z_start = next_shooter_pos[0], next_shooter_pos[2]
    elif next_event['Name'] == 'EventTurnover':
        for person in next_event['Persons']:
            if person['Role'] == 'MapRoleCausedTurnover':
                next_winner_id = person['PersonId']
                break  # Stop searching once we find the winner
        next_winner_pos = get_person_info(next_frame, next_winner_id, 'Position')
        next_x_start, next_z_start = next_winner_pos[0], next_winner_pos[2]
        
    if event['Name'] == 'EventShot':
        if shot_result == 1:
            x_end, z_end = goal_loc[0], goal_loc[2]
        else:
            x_end, z_end = next_x_start, next_z_start

    features['start_x_ai'] = x_start
    features['start_z_ai'] = z_start
    features['end_x_ai'] = x_end
    features['end_z_ai'] = z_end

    # Start- and endpolar of action
    if event['Name'] == 'EventPass':
        start_dist_to_goal_ai, start_angle_to_angle_ai = calculate_distance(passer_pos, goal_loc), polar(passer_pos, goal_loc)
        end_dist_to_goal_ai, end_angle_to_angle_ai = calculate_distance(receiver_pos, goal_loc), polar(receiver_pos, goal_loc)
    elif event['Name'] == 'EventShot':
        start_dist_to_goal_ai, start_angle_to_angle_ai = calculate_distance(shooter_pos, goal_loc), polar(shooter_pos, goal_loc)
        if shot_result == 1:
            end_dist_to_goal_ai, end_angle_to_angle_ai = calculate_distance(goal_loc, goal_loc), polar(goal_loc, goal_loc)
        else:
            end_dist_to_goal_ai, end_angle_to_angle_ai = calculate_distance([next_x_start, 0, next_z_start], goal_loc), polar([next_x_start, 0, next_z_start], goal_loc)
    elif event['Name'] == 'EventTurnover':
        start_dist_to_goal_ai, start_angle_to_angle_ai = calculate_distance(winner_pos, goal_loc), polar(winner_pos, goal_loc)
        end_dist_to_goal_ai, end_angle_to_angle_ai = calculate_distance([next_x_start, 0, next_z_start], goal_loc), polar([next_x_start, 0, next_z_start], goal_loc)

    features['start_dist_to_goal_ai'] = start_dist_to_goal_ai
    features['start_angle_to_angle_ai'] = start_angle_to_angle_ai
    features['end_dist_to_goal_ai'] = end_dist_to_goal_ai
    features['end_angle_to_angle_ai'] = end_angle_to_angle_ai

    # Movement
    # Distance covered + speed
    if event['Name'] == 'EventPass':
        movement = calculate_distance(passer_pos, receiver_pos)
        speed = get_person_info(frame, passer_id, 'Speed')
    elif event['Name'] == 'EventShot':
        speed = get_person_info(frame, shooter_id, 'Speed')
        if shot_result == 1:
            movement = calculate_distance(shooter_pos, goal_loc)
        else: 
            movement = calculate_distance(shooter_pos, [next_x_start, 0, next_z_start])
    elif event['Name'] == 'EventTurnover':
        movement = calculate_distance(winner_pos, [next_x_start, 0, next_z_start])
        speed = get_person_info(frame, winner_id, 'Speed')
    features['movement_ai'] = movement

    # Match Score (MatchScoreContext) - update if present
    goals_home, goals_away = score
    goal_diff = abs(goals_home - goals_away)
    features['goalscore_team'] = goals_home
    features['goalscore_opponent'] = goals_away
    features['goalscore_diff'] = goal_diff
    
    strength_diff = abs(home_strength - visitor_strength)
    features['home_strength'] = home_strength
    features['visitor_strength'] = visitor_strength
    features['strength_difference'] = strength_diff
    
    opp_pos  = init_opp(frame, side)
    shot_cone = [(x_start, z_start), (26.95, -1.2), (26.95, 1.2)]
    total_mass = opponent_density(opp_pos, shot_cone)
    features['defensive_density'] = total_mass
    
    features['speed'] = speed
    
    return features

def build_gameevents_dataframe(event_dict, game_id):
    all_features = []
    home_possession = False

    event_list = list(event_dict.items())
    num_events = len(event_list)

    relevant_actions = ['EventPass', 'EventShot', 'EventTurnover']

    goals_home = 0
    goals_away = 0
    
    home_strength = 6
    visitor_strength = 6

    for i in range(len(event_list)):
        event_id, (frame, event, clock) = event_list[i]
        shot_result = 0  # Reset per event

        # Skip if this event is a goal, but mark the previous shot as the scoring one
        if event['Name'] == 'EventGoal':
            continue
        
        # Test if event contains enough info, if not skip current event
        if event['Name'] == 'EventPass':
            passer_id = event['Persons'][1]['PersonId']
            receiver_id = event['Persons'][2]['PersonId']
            test =  get_person_info(frame, passer_id, 'Position')
            test2 = get_person_info(frame, receiver_id, 'Position')
            if test == None or test2 == None:
                continue 
        elif event['Name'] == 'EventShot':
            for person in event['Persons']:
                if person['Role'] == 'MapRoleShooter':
                    shooter_id = person['PersonId']
                    break  # Stop searching once we find the shooter
            test = get_person_info(frame, shooter_id, 'Position')
            if test == None:
                continue 
        elif event['Name'] == 'EventTurnover':
            for person in event['Persons']:
                if person['Role'] == 'MapRoleCausedTurnover':
                    winner_id = person['PersonId']
                    break  # Stop searching once we find the winner
            test = get_person_info(frame, winner_id, 'Position')
            if test == None:
                continue    
            
        if 'PowerPlayContext' in frame:
            home_strength = frame['PowerPlayContext']['HomeStrength']
            visitor_strength = frame['PowerPlayContext']['VisitorStrength']

        # Track team puck possession
        if event['Name'] == 'EventFaceoff' or event['Name'] == 'EventPossessionChangePlayer':
            for person in event['Persons']:
                if person['Role'] == 'MapRoleWinner' or person['Role'] == 'MapRolePossessionGainer':
                    winner_id = person['PersonId']
                    break  # Stop searching once we find the winner
            try:
                winner_teamside = get_person_info(frame, winner_id, 'TeamSide')
                if winner_teamside == 1:
                    home_possession = True
                else:
                    home_possession = False
                continue
            except UnboundLocalError as e:
                continue
        if event['Name'] == 'EventPossessionChangeTeam':
            home_possession = not home_possession
            continue

        # Initialize shot result
        if 'MatchScoreContext' in frame:
            goals_home = frame['MatchScoreContext']['HomeScore']
            goals_away = frame['MatchScoreContext']['AwayScore']
        score = (goals_home, goals_away)

        if event['Name'] == 'EventShot':
            for j in range(i + 1, min(i + 5, num_events)):
                _, (_, future_event, _) = event_list[j]
                if future_event['Name'] == 'EventShot':
                    break  # Found another shot first → current shot didn't lead to goal
                if future_event['Name'] == 'EventGoal':
                    shot_result = 1  # Current shot led to this goal
                    break
                
        k = i + 1
        while k < num_events:
            _, (frame_k, action_k, _) = event_list[k]

            if action_k['Name'] in relevant_actions:
                next_frame = frame_k
                next_action = action_k
                break  # Found a relevant action

            k += 1  # Move to next event
          
        # Test if next event contains enough info, if not skip current event
        if next_action['Name'] == 'EventPass':
            next_passer_id = next_action['Persons'][1]['PersonId']
            test_next =  get_person_info(next_frame, next_passer_id, 'Position')
            if test_next == None:
                continue             
        elif next_action['Name'] == 'EventShot':
            for person in next_action['Persons']:
                if person['Role'] == 'MapRoleShooter':
                    next_shooter_id = person['PersonId']
                    break  # Stop searching once we find the shooter
            test_next = get_person_info(next_frame, next_shooter_id, 'Position')
            if test_next == None:
                continue               
        elif next_action['Name'] == 'EventTurnover':
            for person in next_action['Persons']:
                if person['Role'] == 'MapRoleCausedTurnover':
                    next_winner_id = person['PersonId']
                    break  # Stop searching once we find the winner
            test_next = get_person_info(next_frame, next_winner_id, 'Position')
            if test_next == None:
                continue               

        features = extract_features(frame, event, clock, home_possession, shot_result, next_frame, next_action, score, game_id, home_strength, visitor_strength)
        all_features.append(features)

    df = pd.DataFrame(all_features)

    # Add time_delta and space_delta
    df['time_delta_i'] = df['time'].diff().shift(-1)
    df['dx_a0i'] = df['start_x_ai'].shift(-1) - df['end_x_ai']
    df['dy_a0i'] = df['start_z_ai'].shift(-1) - df['end_z_ai']
    df['mov_a0i'] = np.hypot(df['dx_a0i'], df['dy_a0i'])

    return df

files_directory = "../../Daan/Output/IceHockey"
processed_directory = "../../processed_data"
output_csv = os.path.join(processed_directory, "all_actions_4.0.csv")

# Get all the filenames
all_files = [f for f in os.listdir(files_directory) if os.path.isfile(os.path.join(files_directory, f))]

for i, filename in enumerate(all_files):
    json_filename = f"{filename}.json"
    json_path = os.path.join(processed_directory, json_filename)
    
    # # Skip file if already processed
    # if os.path.exists(json_path):
    #     print(f"Skipping already processed file: {filename}")
    #     continue

    # Full path to inputfile
    input_path = os.path.join(files_directory, filename)

    # # Process file
    # try:
    #     events = sync_one_game(input_path, i, filename)
    # except (IndexError, json.JSONDecodeError) as e:
    #     continue

    # Load data from processed file
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Skipping missing file: {json_path}")
        continue

    try:
        df = build_gameevents_dataframe(data, i)
    except UnboundLocalError as e:
        continue

    # Write df to csv
    file_exists = os.path.isfile(output_csv)
    df.to_csv(output_csv, mode='a', header=not file_exists, index=False)

end = time.perf_counter()
print(end - start)