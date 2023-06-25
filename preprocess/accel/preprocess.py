import os
import pickle
import pathlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('../..')
import lared.accel.constants.constants as const
from jose.accel.preproc import interpolate
from lared_dataset.constants import (
    raw_accel_path,
    processed_accel_path
)


base_path = pathlib.Path(raw_accel_path)
MAPPING_FILE    = base_path / "mapping.csv"
MASTER_PICKLE_PATH = base_path / "master_data.pkl"
VALID_AUDIO_SEGMENTS_PATH = "../valid_audio_segments.pkl"


balloon_pop_1_video_frame = 23030 # to
balloon_pop_1_accel_frame = 45977 + 19/34

balloon_pop_2_video_frame = 74844
balloon_pop_2_accel_frame = 47706 + 23/28

balloon_pop_3_video_frame = 166836.5
balloon_pop_3_accel_frame = 50776 + 30.5/32

frame_to_accel = interp1d([balloon_pop_1_video_frame, balloon_pop_3_video_frame], [balloon_pop_1_accel_frame, balloon_pop_3_accel_frame], fill_value="extrapolate")
video_seconds_to_accel_sample = interp1d([balloon_pop_1_video_frame/29.97, balloon_pop_3_video_frame/29.97], [balloon_pop_1_accel_frame, balloon_pop_3_accel_frame], fill_value="extrapolate")