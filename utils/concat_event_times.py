import os 
import pandas as pd
from pathlib import Path
import numpy as np
def concat_event_times(source_path, remove_txt_files=False):
    """
    Loads separate event times txt files and organizes them into a df.  
    Saves the df to a csv in the source_path. optionally removes old txt files after completion
    Args:
        source_path: Path where event times txt files are located
        remove_txt_files: Whether to remove the txt files after completion
    Returns:
        df: Combined DataFrame with the event times
    """

    puff_times_path = list(source_path.glob("*5_0_0.txt"))[0] # puff event times
    cue_times_path = list(source_path.glob("*5_1_0.txt"))[0] # cue event times
    rf_times_path = list(source_path.glob("*5_2_0.txt"))[0] # rf event times
    lick_times_path = list(source_path.glob("*5_3_0.txt"))[0] # lick event times
    cam_times_path = list(source_path.glob("*5_4_0.txt"))[0] # cam event times
    # load all txt files into a list of dfs
    try: 
        puff_times_df = pd.read_csv(puff_times_path, sep='\t', header=None, names=['puff_onset'])
    except: # in case there are no events (e.g. during pre exposure)
        puff_times_df = pd.DataFrame([np.nan], columns=['puff_onset'])
        print(f'No airpuff events found for session {source_path.parent.parent.name}')
    try:
        cue_times_df = pd.read_csv(cue_times_path, sep='\t', header=None, names=['cue_onset'])
    except:
        cue_times_df = pd.DataFrame([np.nan], columns=['cue_onset'])
        print(f'No cue events found for session {source_path.parent.parent.name}')
    try:
        rf_times_df = pd.read_csv(rf_times_path, sep='\t', header=None, names=['rf_onset'])
    except:
        rf_times_df = pd.DataFrame([np.nan], columns=['rf_onset'])
        print(f'No RF events found for session {source_path.parent.parent.name}')
    try:
        lick_times_df = pd.read_csv(lick_times_path, sep='\t', header=None, names=['lick_onset'])
    except:
        lick_times_df = pd.DataFrame([np.nan], columns=['lick_onset'])
        print(f'No lick events found for session {source_path.parent.parent.name}')
    try:
        cam_times_df = pd.read_csv(cam_times_path, sep='\t', header=None, names=['cam_start_stop'])
    except:
        cam_times_df = pd.DataFrame([np.nan], columns=['cam_start_stop'])   
        print(f'No cam events found for session {source_path.parent.parent.name}')
    # combine all dfs into a single df
    df = pd.concat([puff_times_df, cue_times_df, rf_times_df, lick_times_df, cam_times_df], axis = 1)
    df.to_csv(source_path / 'event_times.csv', index=False)

    # remove txt files if desired
    if remove_txt_files:
        for file in [puff_times_path, cue_times_path, rf_times_path, lick_times_path, cam_times_path]:
            os.remove(file)
    return df
