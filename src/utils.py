import json

import numpy as np
import pandas as pd


def flatten_l_o_l(nested_list):
    """Flatten a list of lists into a single list.

    Args:
        nested_list (list): 
            – A list of lists (or iterables) to be flattened.

    Returns:
        list: A flattened list containing all items from the input list of lists.
    """
    return [item for sublist in nested_list for item in sublist]


def print_ln(symbol="-", line_len=110, newline_before=False, newline_after=False):
    """Print a horizontal line of a specified length and symbol.

    Args:
        symbol (str, optional): 
            – The symbol to use for the horizontal line
        line_len (int, optional): 
            – The length of the horizontal line in characters
        newline_before (bool, optional): 
            – Whether to print a newline character before the line
        newline_after (bool, optional): 
            – Whether to print a newline character after the line
    """
    if newline_before: print();
    print(symbol * line_len)
    if newline_after: print();


def read_json_file(file_path):
    """Read a JSON file and parse it into a Python object.

    Args:
        file_path (str): The path to the JSON file to read.

    Returns:
        dict: A dictionary object representing the JSON data.
        
    Raises:
        FileNotFoundError: If the specified file path does not exist.
        ValueError: If the specified file path does not contain valid JSON data.
    """
    try:
        # Open the file and load the JSON data into a Python object
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        return json_data
    except FileNotFoundError:
        # Raise an error if the file path does not exist
        raise FileNotFoundError(f"File not found: {file_path}")
    except ValueError:
        # Raise an error if the file does not contain valid JSON data
        raise ValueError(f"Invalid JSON data in file: {file_path}")


def get_sign_df(pq_path, invert_y=True):
    sign_df = pd.read_parquet(pq_path)

    # y value is inverted (Thanks @danielpeshkov)
    if invert_y: sign_df["y"] *= -1

    return sign_df


def load_relevant_data_subset(pq_path, CFG):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / CFG.ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, CFG.ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


def title_map_fn(ann):
    title_map = {
        'face_nan_pct': '<b>Percentage Of <i>Face</i> Data Points That Are NaN</b>', 
        'left_hand_nan_pct': '<b>Percentage Of <i>Left Hand</i> Data Points That Are NaN</b>',
        'pose_nan_pct': '<b>Percentage Of <i>Pose</i> Data Points That Are NaN</b>',
        'right_hand_nan_pct': '<b>Percentage Of <i>Right Hand</i> Data Points That Are NaN</b>'}
    ann.text = title_map.get(ann.text[1:])
    return ann
