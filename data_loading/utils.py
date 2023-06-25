import csv
import os

import pickle

import json

import numpy as np
import pandas as pd


# csv_path = "../preprocess/audio/vad/"

# csv file path for successful training dataset.
from scipy.interpolate import interp1d

csv_path = "../preprocess/audio/successful_train_samples/"

# csv file path for successful testing dataset.
test_csv_path = "../preprocess/audio/successful_test_samples/"

# csv file of all_unsuccessful unsucessful testing samples including both start and continue category.
test_csv_path_unsuccessful = "../preprocess/audio/unsuccessful_intention_test_sample/"

# csv file of all_unsuccessful unsucessful testing samples including both start and continue category.
test_csv_path_all_sample = "../preprocess/audio/all_sample/"

# txt output file of customized check
txt_path = "../preprocess/audio/output_file/"



balloon_pop_1_video_frame = 23030 # to
balloon_pop_1_accel_frame = 45977 + 19/34

balloon_pop_2_video_frame = 74844
balloon_pop_2_accel_frame = 47706 + 23/28

balloon_pop_3_video_frame = 166836.5
balloon_pop_3_accel_frame = 50776 + 30.5/32


def reset_examples_ids(examples):
    for i, ex in enumerate(examples):
        ex['id'] = i


'''
Generate information for example.pkl file
'''


def get_words(pid, ini_time, end_time):
    # Load the json file for the current participant
    with open(f'../wordLib{pid}.json', 'r') as f:
        word_data = json.load(f)

    # Filter the words that fall within the time window
    words_in_window = [entry[0] for entry in word_data if ini_time <= entry[1] <= end_time]

    return words_in_window

class Maker():
    def __init__(self, accel_path=None, vad_path=None, unsuccessful_vad_path=None, all_sample_path=None):

        self.accel = {}
        if accel_path is not None:
            self.load_accel(accel_path)

        self.vad = {}
        if vad_path is not None:
            self.load_vad(vad_path)

        self.unsuccessful_vad = {}
        if unsuccessful_vad_path is not None:
            self.load_unsuccessful_vad(unsuccessful_vad_path)

        self.all_samples_vad = {}
        if all_sample_path is not None:

            self.load_all_vad(all_sample_path)

        self.examples = None
        self.test_examples = None
        self.unsuccessful_examples = None
        self.all_samples = None

    def load_accel(self, accel_path):
        self.accel = pickle.load(open('../average_vectors.pkl', 'rb'))

        # # self.accel = pickle.load(open(accel_path, 'rb'))
        # self.accel = {}
        # if accel_path is not None:
        # #self.accel = pickle.load(open('../data/subj_accel_interp.pkl', 'rb'))
        #     self.accel = self.load_accel(accel_path)

    def load_vad(self, vad_path):

        # load csv directly
        self.vad = {}
        pid_list = [2, 3, 4, 5, 7, 10, 11, 17, 22, 23, 27, 34, 35]
        for i in pid_list:
            fpath = os.path.join(vad_path, f'{i}.csv')

            self.vad[i] = pd.read_csv(fpath, header=None).to_numpy()

        if len(self.vad) == 0:
            print('load_vad called but nothing loaded.')

    def load_unsuccessful_vad(self, unsuccessful_vad):
        start_pid = [2, 3, 4, 5, 7, 10, 11, 17, 22, 23, 27, 34, 35]
        for i in start_pid:
            fpath = os.path.join(unsuccessful_vad, f'{i}.csv')
            self.unsuccessful_vad[i] = pd.read_csv(fpath, header=None).to_numpy()

        if len(self.unsuccessful_vad) == 0:
            print('load_unsuccessful_vad called but nothing loaded.')

    def _interp_vad(self, vad, in_fs, out_fs):
        t = np.arange(0, len(vad) / in_fs, 1/in_fs)
        f = interp1d(t, vad, kind='nearest')
        tnew = np.arange(0, len(vad) / in_fs, 1/out_fs)
        return f(tnew)

    def load_all_vad(self, all_vad):
        pid_list = [2, 3, 4, 5, 7, 10, 11, 17, 22, 23, 27, 34, 35]
        continue_pid = []
        maybe_pid = []
        for i in pid_list:
            fpath = os.path.join(all_vad, f'{i}.csv')

            self.all_samples_vad[i] = pd.read_csv(fpath, header=None).to_numpy()


        if len(self.all_samples_vad) == 0:
            print('load_all_vad called but nothing loaded.')

    # set time window
    def _get_vad(self, pid, ini_time, end_time, fs):
        # note audio (vad) and video start at the same time
        if pid not in self.vad:
            return None

        return self.vad[pid][ini_time * fs: end_time * fs].flatten()

    def _get_unsuccessful_vad(self, pid, ini_time, end_time, fs):
        # note audio (vad) and video start at the same time
        if pid not in self.unsuccessful_vad:
            return None

        return self.unsuccessful_vad[pid][ini_time * fs: end_time * fs].flatten()

    def _get_all_vad(self, pid, ini_time, end_time, fs):
        # note audio (vad) and video start at the same time
        if pid not in self.all_samples_vad:
            return None

        return self.all_samples_vad[pid][ini_time * fs: end_time * fs].flatten()

    def make_train_examples(self, windowSize, feature_fs):

        examples = list()
        example_id = 0
        valid_list = [2, 3, 4, 5, 7, 10, 11, 17, 22, 23, 27, 34, 35]

        # loop over participants
        for i in valid_list:

            #print("pid : ", i)

            time_window_list = []
            # read train participant csv
            # generate training dataset time window of successful intention case
            with open(csv_path + str(windowSize) + "s/" + '_' + str(i) + ".csv") as infile:
                reader = csv.reader(infile)

                for line in reader:
                    if line:
                        time_window_list.append(tuple([int(line[0]), int(line[1])]))

            # create dict for successful intention training dataset.
            for j in range(0, len(time_window_list)):
                ini_time = time_window_list[j][0]
                end_time = time_window_list[j][1]

                temp_vad = self._get_vad(i, ini_time, end_time, 100)
                interp_vad = self._interp_vad(temp_vad, 100, feature_fs)

                # Convert times to indices (if your times are not already in seconds)
                start_index = int(ini_time)
                end_index = int(end_time)

                # Get the relevant slice of the array
                vectors = self.accel[i][start_index:end_index]

                vectors = vectors.transpose().astype(np.float32)

                words = get_words(i, ini_time, end_time)

                examples.append({
                    'id': example_id,
                    'pid': i,
                    'ini_time': ini_time,
                    'end_time': end_time,
                    'data': vectors,
                    'vad': interp_vad,
                    'words': words
                })
                example_id += 1

        self.examples = examples

        return examples

    def make_test_examples(self, index_s, windowSize, feature_fs):
        test_examples = list()
        test_example_id = 0

        valid_list = [2, 3, 4, 5, 7, 10, 11, 17, 22, 23, 27, 34, 35]

        # loop over participants
        for i in valid_list:


            # generate test dataset time window of successful intention case:
            test_time_window_list = []
            with open(test_csv_path +  str(windowSize) + "s/" + str(index_s) + '_' + str(i) + ".csv") as infile:
                test_reader = csv.reader(infile)

                for test_line in test_reader:
                    if test_line:
                        test_time_window_list.append(tuple([int(test_line[0]), int(test_line[1])]))

            # create dict for successful intention testing dataset.
            for j in range(0, len(test_time_window_list)):

                test_ini_time = test_time_window_list[j][0]
                test_end_time = test_time_window_list[j][1]

                test_temp_vad = self._get_vad(i, test_ini_time, test_end_time, 100)

                interp_vad = self._interp_vad(test_temp_vad, 100, feature_fs)

                # Convert times to indices (if your times are not already in seconds)
                start_index = int(test_ini_time)
                end_index = int(test_end_time)


                # Get the relevant slice of the array
                vectors = self.accel[i][start_index:end_index]

                vectors = vectors.transpose().astype(np.float32)

                words = get_words(i, test_ini_time, test_end_time)

                test_examples.append({
                    'words': words,
                    'id': test_example_id,
                    'pid': i,
                    'ini_time': test_ini_time,
                    'end_time': test_end_time,
                    'data': vectors,
                    'vad': interp_vad
                })
                test_example_id += 1
        self.test_examples = test_examples

        return test_examples

    def make_all_examples(self, index_s, windowSize, feature_fs):
        """
        :param unsuccessful_pid: pid number
        :param index_s: the number of experiment
        :return: all_unsuccessful testing dataset including both start and continue unsuccessful case.
        """
        examples = list()
        example_id = 0
        valid_list = [2, 3, 4, 5, 7, 10, 11, 17, 22, 23, 27, 34, 35]

        for i in valid_list:
            all_test_time_window_list = []

            with open(test_csv_path_all_sample +  str(windowSize) + "s/" + str(index_s) + '_' + str(i) + ".csv") as infile:
                all_reader = csv.reader(infile)

                for all_sample_line in all_reader:
                    if all_sample_line:
                        all_test_time_window_list.append(tuple([int(all_sample_line[0]),
                                                                int(all_sample_line[1])]))

            for j in range(0, len(all_test_time_window_list)):
                all_ini_time = all_test_time_window_list[j][0]
                all_end_time = all_test_time_window_list[j][1]

                all_vad = self._get_all_vad(i, all_ini_time, all_end_time, 100)

                interp_vad = self._interp_vad(all_vad, 100, feature_fs)

                start_index = int(all_ini_time)
                end_index = int(all_end_time)

                # Get the relevant slice of the array
                vectors = self.accel[i][start_index:end_index]

                vectors = vectors.transpose().astype(np.float32)

                words = get_words(i, all_ini_time, all_end_time)

                examples.append({
                    'id': example_id,
                    'pid': i,
                    'ini_time': all_ini_time,
                    'end_time': all_end_time,
                    'data' : vectors,
                    'vad': interp_vad,
                    'words' : words
                })
                example_id += 1

        self.examples = examples
        return examples

    def make_unsuccessful_examples(self, unsuccessful_pid, index_s, windowSize, feature_fs, category:str):
        """
        :param unsuccessful_pid: pid number
        :param index_s: the number of experiment
        :return: all_unsuccessful testing dataset including both start and continue unsuccessful case.
        """
        examples = list()
        example_id = 0
        valid_list = [2, 3, 4, 5, 7, 10, 11, 17, 22, 23, 27, 34, 35]
        for i in valid_list:
            all_test_time_window_list = []

            if unsuccessful_pid.__contains__(i):

                with open(test_csv_path_unsuccessful + category + "/" + str(windowSize) +"s/" + str(index_s) + '_' + str(i) + ".csv") as infile:
                    all_reader = csv.reader(infile)

                    for unsuccessful_line in all_reader:
                        if unsuccessful_line:
                            all_test_time_window_list.append(tuple([int(unsuccessful_line[0]),
                                                                    int(unsuccessful_line[1])]))

                for j in range(0, len(all_test_time_window_list)):

                    all_ini_time = all_test_time_window_list[j][0]
                    all_end_time = all_test_time_window_list[j][1]

                    unsuccessful_temp_vad = self._get_unsuccessful_vad(i, all_ini_time, all_end_time, 100)

                    interp_vad = self._interp_vad(unsuccessful_temp_vad, 100, feature_fs)

                    start_index = int(all_ini_time)
                    end_index = int(all_end_time)

                    # Get the relevant slice of the array
                    vectors = self.accel[i][start_index:end_index]

                    vectors = vectors.transpose().astype(np.float32)

                    words = get_words(i, all_ini_time, all_end_time)

                    examples.append({
                        'id': example_id,
                        'pid': i,
                        'ini_time': all_ini_time,
                        'end_time': all_end_time,
                        'data': vectors,
                        'words': words,
                        'vad': interp_vad
                    })

                    example_id += 1

        self.examples = examples
        return examples

    def filter_examples_by_movement_threshold(self, ts=20):
        new_examples = list()

        for ex in self.examples:
            track = ex['poses']
            std_x = np.std(track[:, 3])
            std_y = np.std(track[:, 4])

            if std_x > ts or std_y > ts:
                continue

            new_examples.append(ex)


video_seconds_to_accel_sample = interp1d(
    [
        balloon_pop_1_video_frame/29.97,
        balloon_pop_3_video_frame/29.97
    ], [
        balloon_pop_1_accel_frame,
        balloon_pop_3_accel_frame
    ], fill_value="extrapolate")