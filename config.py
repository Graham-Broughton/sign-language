import os
from dataclasses import dataclass

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, 'data')
raw_data_path = os.path.join(data_path, 'raw')
landmarks = os.path.join(raw_data_path, 'train_landmark_files')
pred_idx = os.path.join(raw_data_path, 'sign_to_prediction_index_map.json')


@dataclass
class CFG:
    BASE_PATH: str = base_path
    DATA_PATH: str = data_path
    RAW_DATA_PATH: str = raw_data_path
    LANDMARKS: str = landmarks
    PREDICTION_IDX: str = pred_idx
    SEED: int = 42
    ROWS_PER_FRAME: int = 543
    EXAMINE_PCT: float = 0.001
    