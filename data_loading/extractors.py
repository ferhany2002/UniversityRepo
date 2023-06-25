
import pickle

import numpy as np

from scipy.interpolate import interp1d

balloon_pop_1_video_frame = 23030 # to
balloon_pop_1_accel_frame = 45977 + 19/34

balloon_pop_2_video_frame = 74844
balloon_pop_2_accel_frame = 47706 + 23/28

balloon_pop_3_video_frame = 166836.5
balloon_pop_3_accel_frame = 50776 + 30.5/32

class AccelExtractor():
    def __init__(self, accel_path, strict=False):
        self.accel = pickle.load(open(accel_path, 'rb'))
        self.strict = strict

        self.num_channels = 3
        self.fs = 20

    def __call__(self, pid, start, end):


        if self.strict and pid not in self.accel:
            raise ValueError(f'pid {pid} not in self.accel')
        
        if pid not in self.accel:
            return np.zeros((self.num_channels, round(self.fs * (end-start))), dtype=np.float32)

        accel_ini = video_seconds_to_accel_sample(start)

        accel_fin = video_seconds_to_accel_sample(end)



        my_subj_accel = self.accel[pid]
        temp = my_subj_accel[:,0] > accel_ini

        ini_idx = np.argmax(my_subj_accel[:,0] > accel_ini)
        fin_idx = ini_idx + round(self.fs * (end-start))


        accel = my_subj_accel[ini_idx: fin_idx, 1:]
        if len(accel) < round(self.fs * (end-start)):
            accel = np.pad(accel, 
                ((0, round(self.fs * (end-start))-len(accel)), (0, 0)),
                mode='constant',
                constant_values= 0)

        return accel.transpose().astype(np.float32)

    def extract_multiple(self, keys):
        return np.stack([self(*k) for k in keys])


video_seconds_to_accel_sample = interp1d(
    [
        balloon_pop_1_video_frame/29.97,
        balloon_pop_3_video_frame/29.97
    ], [
        balloon_pop_1_accel_frame,
        balloon_pop_3_accel_frame
    ], fill_value="extrapolate")