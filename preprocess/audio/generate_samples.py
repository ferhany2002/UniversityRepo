import csv
import operator

import random as rand


import pandas as pd
from scipy.io import wavfile


"""
This file is use for generating training samples and corresponding ground truth.
"""

diarizations_path = './rttmFile'

# path for vad files.
vad_path = 'filter_vad/'

# global variable
pid_list = [2, 3, 4, 5, 7, 10, 11, 17, 22, 23, 27, 34, 35]  # len = 13
start_pid = [2, 3, 4, 5, 7, 10, 11, 17, 22, 23, 27, 34, 35]
train_speech_total = 0
train_silence_total = 0
test_speech_total = 0
test_silence_total = 0
total_speech = 0
total_silence = 0


def load_diarization(fpath):
    return pd.read_csv(fpath,
                       header=None,
                       names=['x', 'y', 'z', 'ini', 'dur', 'n1', 'n2', 'speaker', 'n3', 'n4'],
                       usecols=['ini', 'dur', 'speaker'],
                       delim_whitespace=True,
                       index_col=False)


# d = load_diarization(os.path.join(diarizations_path, '7.rttm'))
# row = d.iloc[7, :]

main_speakers = {
    1: [0],
    2: [0],
    3: [0],
    4: [0],
    5: [0],
    7: [1, 3],
    9: [0],
    10: [1],
    11: [0],
    12: [0, 1],
    13: [1],
    14: [1],
    15: [0],
    16: [0],
    17: [0],
    18: [0],  # fail: two women with same voice
    19: [0],
    20: [0],
    21: [1],
    22: [0],
    23: [1],
    24: [0],
    25: [1],
    26: [0],
    27: [2],
    29: [1, 3],  # check
    30: [0, 3],  # check
    31: [0],
    32: [0],
    33: [0],
    34: [0],
    35: [1],
    45: [0]
}

# start_pid = [2,3,4,7,10,11,17,22,23, 34]

"""
Manually labeled data
time frame of unrealized intention
"""
unsuccessful_intention = [(2, 385570, 387670, 'INTS_start'), (2, 326840, 328570, 'INTS_continue'),
(3, 139500, 141950, 'INTS_continue'), (3, 371950, 374520, 'INTS_continue'), (4, 127110, 129170, 'INTS_start'),
                            (4, 44860, 47690, 'INTS_continue'), (4, 192490, 194400, 'INTS_start'),
                            (4, 428950, 431620, 'INTS_start'), (4, 454360, 456800, 'INTS_start'),
                            (4, 461680, 464710, 'INTS_start'), (4, 112740, 115090, 'INTS_start'),
                            (4, 196910, 198870, 'INTS_start'), (4, 266310, 267900, 'INTS_continue'),
                            (4, 284290, 286930, 'INTS_continue'), (4, 297150, 298820, 'INTS_start'),
                            (4, 310960, 312890, 'INTS_continue'), (5, 397280, 400500, 'INTS_continue'),
                            (5, 245770, 248370, 'INTS_start'), (5, 412940, 414980, 'INTS_start'),
                            (7, 317660, 319900, 'INTS_continue'), (7, 570090, 574000, 'INTS_start'),
                            (10, 265920, 267680, 'INTS_continue'), (10, 268070, 270040, 'INTS_continue'),
                            (10, 299760, 301570, 'INTS_continue'), (10, 322010, 323890, 'INTS_start'),
                            (10, 506090, 507620, 'INTS_continue'), (11, 35322, 36322, 'INTS_start'),
                            (11, 569646, 572020, 'INTS_start'), (11, 50231, 51231, 'INTS_start'), (11, 52129, 53129, 'INTS_start'),
                            (11, 284610, 285610, 'INTS_continue'), (11, 504946, 505946, 'INTS_start'),
                            (17, 127110, 128590, 'INTS_continue'), (17, 234790, 236770, 'INTS_continue'),
                            (17, 586720, 588630, 'INTS_start'), (22, 319730, 322010, 'INTS_start'),
                            (22, 245900, 248380, 'INTS_start'), (22, 355920, 359270, 'INTS_start'),
                            (23, 543100, 545700, 'INTS_continue'),
                            (22, 531540, 536200, 'INTS_continue'), (22, 555410, 557510, 'INTS_start'),
                            (27, 472000, 476363, 'INTS_start'), (27, 35363, 38454, 'INTS_continue'),
                            (27, 44909, 48909, 'INTS_start'), (27, 86181, 88090, 'INTS_continue'),
                            (34, 426545, 432090, 'INTS_start'), (34, 439545, 440727, 'INTS_start'),
                            (34, 473363, 478545, 'INTS_start'), (34, 515636, 518636, 'INTS_start'),
                            (35, 214080, 216580, 'INTS_start')]

# produce VAD

import numpy as np
from scipy.io.wavfile import write as wavwrite


# exclue 3600 - 4200 seconds (1:00 h - 1:10 h)
def generate_negative_sample(vad, train_intention_time_window, test_intention_time_window, intention_label, duration,
                             ratio,
                             size=9900,
                             fs=100):
    """
    generate negative samples for successful intention case.

    : param p1: vad file
    : param p2: array of positive training samples
    : param p3: array of positive testing samples
    : param p4: label array corresponding to the positive samples
    : param p5: window size
    : param p6: ratio of positive samples to negative samples.
    : param p7: array size of original vad file
    : param p8: sampling frequency

    : return r1: array of negatives training samples
    : return r2: array of negative testing samples
    : return r3: training samples overlap with speech
    : return r4: training samples not overlap with speech
    : return r5: testing samples overlap with speech
    : return r6: testing samples not overlap with speech

    """
    # negative_intention_label = np.zeros((size * fs))
    train_negative_time_sample_size = len(train_intention_time_window) * ratio
    test_negative_time_sample_size = len(test_intention_time_window)
    negative_intention_time_window_list_train = []
    negative_intention_time_window_list_test = []
    # 3600 - 4200 seconds (1:00 h - 1:10 h)
    false_negative_check_time_window = []
    collect_false_negative = 3

    train_speech_overlap = 0
    train_silence = 0
    test_speech_overlap = 0
    test_silence = 0

    while len(negative_intention_time_window_list_train) < train_negative_time_sample_size or len(
            negative_intention_time_window_list_test) < test_negative_time_sample_size:

        random_point = (rand.randint(0, 9900))

        # check left side of random point
        left_point = random_point - duration
        if left_point >= 0:

            if  not intention_label[left_point * fs:random_point * fs].__contains__(1):

                # exclude time interval of 3600 - 4200  for training dataset
                if (left_point > 4200 or random_point < 3600) and len(
                        negative_intention_time_window_list_train) < train_negative_time_sample_size:
                    negative_intention_time_window_list_train.append(tuple([left_point, random_point]))

                    if vad[left_point * fs: random_point * fs].__contains__(1):
                        train_speech_overlap += 1
                    else:
                        train_silence += 1

                # during time interval of 3600 - 4200 for test dataset
                if left_point >= 3600 and random_point <= 4200 and len(
                        negative_intention_time_window_list_test) < test_negative_time_sample_size:
                    negative_intention_time_window_list_test.append(tuple([left_point, random_point]))

                    if vad[left_point * fs: random_point * fs].__contains__(1):
                        test_speech_overlap += 1
                    else:
                        test_silence += 1

        # check right side of random point
        right_point = random_point + duration

        if right_point <= 9900:

            if not intention_label[random_point * fs:right_point * fs].__contains__(1):

                # exclude time interval of 3600 - 4200
                if (random_point > 4200 or right_point < 3600) and len(
                        negative_intention_time_window_list_train) < train_negative_time_sample_size:
                    negative_intention_time_window_list_train.append(tuple([random_point, right_point]))

                    if vad[random_point * fs: right_point * fs].__contains__(1):
                        train_speech_overlap += 1
                    else:
                        train_silence += 1

                # during time interval of 3600 - 4200
                # print("in test nagetive sample")
                if random_point >= 3600 and right_point <= 4200 and len(
                        negative_intention_time_window_list_test) < test_negative_time_sample_size:
                    negative_intention_time_window_list_test.append(tuple([random_point, right_point]))

                    if vad[random_point * fs: right_point * fs].__contains__(1):
                        test_speech_overlap += 1
                    else:
                        test_silence += 1

    return negative_intention_time_window_list_train, negative_intention_time_window_list_test, train_speech_overlap, train_silence, test_speech_overlap, test_silence


# exclue 3600 - 4200 seconds (1:00 h - 1:10 h)
def generate_positive_sample(vad, duration, size=9900, fs=100):
    """
    generate positive samples for successful intention case

    : param p1: vad file
    : param p2: window size
    : param p3: full size of the original val file
    : param p4: sampling frequency

    : return r1: array of positive training samples
    : return r2: array of positive testing samples
    : return r3: label array corresponding to the positive samples.
    """

    valid = True
    train_intention_time_window = []
    test_intention_time_window = []
    intention_label = np.zeros((size * fs))
    unique, counts = np.unique(intention_label, return_counts=True)
    # print(dict(zip(unique, counts)))
    previous_is_zero = True

    min_speech_frames = 200
    for i in range(0, size):
        if i + duration < size:

            if (vad[i * fs] == 1) and previous_is_zero:
                speech_frames = 0  # reset the speech frames counter
                previous_is_zero = False

                # check valid time window
                for j in range(i * fs, (i + duration) * fs):
                    if vad[j] == 1:
                        speech_frames += 1  # increment the counter if speech is detected
                    else:
                        valid = False
                        break

                # check minimum speech period
                if speech_frames < min_speech_frames:
                    valid = False

                # intention window
                if valid:
                    time_ini = i
                    time_end = i + duration

                    # exclude time interval of 3600 - 4200
                    if time_ini > 4200 or time_end < 3600:
                        train_intention_time_window.append(tuple([time_ini, time_end]))

                    # during time interval of 3600 - 4200
                    if time_ini >= 3600 and time_end <= 4200:
                        # print("in test positive sample")
                        test_intention_time_window.append(tuple([time_ini, time_end]))

                    intention_label[time_ini * fs:time_end * fs] = 1

            if vad[i * fs] == 0 and not previous_is_zero:
                previous_is_zero = True

            valid = True

    # print("count 1: ", len(train_intention_time_window), " count 1 test : ", len(test_intention_time_window))

    return train_intention_time_window, test_intention_time_window, intention_label


def generate_training_sample(vad, duration, ratio, size=9900, fs=100):
    valid = True
    train_intention_time_window = []
    intention_label = np.zeros((size * fs))
    previous_is_zero = True

    for i in range(0, size):
        if i + duration < size:  # change from i - duration >= 0 to i + duration < size

            if (vad[i * fs] == 1) and previous_is_zero:
                previous_is_zero = False
                # check valid time window
                for j in range(i * fs, (i + duration) * fs):  # change from (i - 1) * fs, (i - duration - 1) * fs, -1 to i * fs, (i + duration) * fs
                    if vad[j] == 0:  # change from vad[j] == 1 to vad[j] == 0
                        valid = False
                        break

                if valid:
                    time_ini = i  # change from i - duration to i
                    time_end = i + duration  # change from i to i + duration

                    # exclude time interval of 3600 - 4200
                    if time_ini > 4200 or time_end < 3600:
                        train_intention_time_window.append(tuple([time_ini, time_end]))


                    intention_label[time_ini * fs:time_end * fs] = 1

            if vad[i * fs] == 0 and not previous_is_zero:
                previous_is_zero = True

            valid = True

    # print("count 1: ", len(train_intention_time_window), " count 1 test : ", len(test_intention_time_window))

    train_negative_time_sample_size = len(train_intention_time_window) * ratio

    negative_intention_time_window_list_train = []
    # 3600 - 4200 seconds (1:00 h - 1:10 h)

    train_speech_overlap = 0
    train_silence = 0

    while len(negative_intention_time_window_list_train) < train_negative_time_sample_size:

        random_point = (rand.randint(0, 9900))

        # check left side of random point
        left_point = random_point - duration
        if left_point >= 0:

            if not vad[left_point * fs:random_point * fs].__contains__(1) and intention_label[left_point * fs:random_point * fs].__contains__(1):

                # exclude time interval of 3600 - 4200  for training dataset
                if (left_point > 4200 or random_point < 3600) and len(
                        negative_intention_time_window_list_train) < train_negative_time_sample_size:
                    negative_intention_time_window_list_train.append(tuple([left_point, random_point]))

                    if vad[left_point * fs: random_point * fs].__contains__(1):
                        train_speech_overlap += 1
                    else:
                        train_silence += 1


        # check right side of random point
        right_point = random_point + duration

        if right_point <= 9900:

            if vad[random_point * fs:right_point * fs].__contains__(1) and not intention_label[random_point * fs:right_point * fs].__contains__(1):

                # exclude time interval of 3600 - 4200
                if (random_point > 4200 or right_point < 3600) and len(
                        negative_intention_time_window_list_train) < train_negative_time_sample_size:
                    negative_intention_time_window_list_train.append(tuple([random_point, right_point]))

                    if vad[random_point * fs: right_point * fs].__contains__(1):
                        train_speech_overlap += 1
                    else:
                        train_silence += 1

    all_train_sample = train_intention_time_window + negative_intention_time_window_list_train
    np.random.shuffle(all_train_sample)

    return all_train_sample, intention_label

def generate_successful_test_sample(vad, duration, ratio, size=9900, fs=100):
    valid = True
    test_intention_time_window = []
    successful_test_label = np.zeros((size * fs))
    previous_is_zero = True
    for i in range(0, size):
        if i + duration < size:  # Ensure we don't go beyond the size
            if (vad[i * fs] == 1) and previous_is_zero:
                previous_is_zero = False
                # check valid time window
                for j in range((i + 1) * fs, (i + duration + 1) * fs):
                    if vad[j] == 0:
                        valid = False
                        break

                # intention 2 seconds window (2 * 100)
                if valid:
                    time_ini = i
                    time_end = i + duration

                    # during time interval of 3600 - 4200
                    if time_ini >= 3600 and time_end <= 4200:
                        # print("in test positive sample")
                        test_intention_time_window.append(tuple([time_ini, time_end]))

                    successful_test_label[time_ini * fs:time_end * fs] = 1

            if vad[i * fs] == 0 and not previous_is_zero:
                previous_is_zero = True

            valid = True

    # print("count 1: ", len(train_intention_time_window), " count 1 test : ", len(test_intention_time_window))

    # negative_intention_label = np.zeros((size * fs))
    test_negative_time_sample_size = len(test_intention_time_window) * ratio
    negative_intention_time_window_list_test = []

    test_speech_overlap = 0
    test_silence = 0

    while len(negative_intention_time_window_list_test) < test_negative_time_sample_size:

        random_point = (rand.randint(0, 9900))

        # check left side of random point
        left_point = random_point - duration
        if left_point >= 0:

            if vad[left_point * fs:random_point * fs].__contains__(1) and not successful_test_label[left_point * fs:random_point * fs].__contains__(1):

                # during time interval of 3600 - 4200 for test dataset
                if left_point >= 3600 and random_point <= 4200 and len(
                        negative_intention_time_window_list_test) < test_negative_time_sample_size:
                    negative_intention_time_window_list_test.append(tuple([left_point, random_point]))

                    if vad[left_point * fs: random_point * fs].__contains__(1):
                        test_speech_overlap += 1
                    else:
                        test_silence += 1

        # check right side of random point
        right_point = random_point + duration

        if right_point <= 9900:

            if vad[random_point * fs:right_point * fs].__contains__(1) and  not successful_test_label[random_point * fs:right_point * fs].__contains__(1):
                # during time interval of 3600 - 4200
                # print("in test nagetive sample")
                if random_point >= 3600 and right_point <= 4200 and len(
                        negative_intention_time_window_list_test) < test_negative_time_sample_size:
                    negative_intention_time_window_list_test.append(tuple([random_point, right_point]))

                    if vad[random_point * fs: right_point * fs].__contains__(1):
                        test_speech_overlap += 1
                    else:
                        test_silence += 1

    all_test_sample = test_intention_time_window + negative_intention_time_window_list_test
    np.random.shuffle(all_test_sample)

    return all_test_sample, successful_test_label




# separate unsuccessful intentions with different types
def preprocess_unsuccessful_intention(ints):
    """
    :param ints: manually labeled data
    :return: three types of manually labeled data
    """
    ints_start = []
    ints_continue = []
    ints_maybe = []
    ints_all = []
    for i in range(len(ints) - 1, -1, -1):
        # delete participant 18
        if ints[i][0] == 18:
            ints.remove(ints[i])
        # filter ints start
        elif ints[i][3] == 'INTS_start':
            ints_start.append(ints[i])
            ints_all.append(ints[i])
        # filter ints continue
        elif ints[i][3] == 'INTS_continue':
            ints_continue.append(ints[i])
            ints_all.append(ints[i])
        # filter ints maybe
        elif ints[i][3] == 'INTS_maybe':
            ints_maybe.append((ints[i]))


    # print(len(ints))
    # print('INTS start number: ', len(ints_start))
    # print(ints_start)
    # print('INTS continue number: ', len(ints_continue))
    # print(ints_continue)
    # print('INTS maybe number: ', len(ints_maybe))
    # print(ints_maybe)

    return ints_start, ints_continue, ints_maybe, ints_all


# generate unsuccessful intentions as labels
def generate_unsuccessful_intention_samples(pid, ints_list, duration, size=9900, fs=100):
    """
    generate positive samples for unsuccessful intention case.

    :param pid: speaker id
    :param ints_list: labeled samples
    :param duration: window size
    :param size: array size of original vad file
    :param fs: sampling frequency
    :return r1: array of positive unsuccessful intention label
    :return r2: array of positive unsuccessful samples
    """

    # initialize ints label array
    ints_ground_truth = np.zeros((size * fs))
    unsuccessful_sample = []

    for i in range(len(ints_list)):
        # print('current ints list ', ints_list[i])
        if pid == ints_list[i][0]:
            # generate positive
            # get ints start time and end time from intention list
            real_start_time = ints_list[i][1] // 1000 + 3600
            end_time = ints_list[i][2] // 1000 + 3600
            # use end time minus time window to create labels
            start_time = end_time - duration
            # set ground truth for unsuccessful intentions with real intention time
            ints_ground_truth[real_start_time * fs:end_time * fs] = 1
            # set positive unsuccessful intentions with end time-2 to end time as positive samples
            unsuccessful_sample.append(tuple([start_time, end_time]))

    # print('ints positive samples are: ', unsuccessful_sample)

    # count_positive = 0
    # for j in range(len(ints_ground_truth)):
    #     if ints_ground_truth[j] == 1:
    #         count_positive += 1
    # print('counting total ones: ', count_positive)

    # ints_ground_truth as array[time*fs] with 1s in unsuccessful intentions
    # ints_positive_sample as list of 2-second ints interval
    return ints_ground_truth, unsuccessful_sample


def generate_negative_unsuccessful_sample(vad, positive_sample, successful_ground_truth, ground_truth, duration, ratio,
                                          fs=100):
    """
    Function to generate unsuccessful intention negative samples.

    :param vad: original vad file
    :param positive_sample: array of positive unsuccessful intention samples
    :param successful_ground_truth: array of successful intention label
    :param ground_truth: array of unsuccessful intention labels
    :param duration: window size
    :param ratio: ratio of positive samples to negative samples.
    :param fs: sampling frequency
    :return r1: array of negative unsuccessful samples
    :return r2: number of sampels overlap with speech
    :return r3: number of samples not overlap with speech
    """

    negative_sample = []
    number_of_negative_sample = len(positive_sample) * ratio

    speech = 0
    silence = 0

    while len(negative_sample) < number_of_negative_sample:
        random_point = rand.randint(0, 600)
        # check left side of random point
        left_point = random_point - duration

        if left_point >= 0:
            if not ground_truth[(left_point + 3600) * fs:(random_point + 3600) * fs].__contains__(
                    1) and not successful_ground_truth[
                               (left_point + 3600) * fs:(random_point + 3600) * fs].__contains__(1):
                negative_sample.append(tuple([left_point + 3600, random_point + 3600]))

                if vad[(left_point + 3600) * fs:(random_point + 3600) * fs].__contains__(1):
                    speech += 1
                else:
                    silence += 1

        # check right side of random point
        right_point = random_point + duration

        if right_point <= 600:
            if not ground_truth[(random_point + 3600) * fs: (right_point + 3600) * fs].__contains__(
                    1) and not successful_ground_truth[
                               (random_point + 3600) * fs: (right_point + 3600) * fs].__contains__(1):
                negative_sample.append(tuple([random_point + 3600, right_point + 3600]))

                if vad[(random_point + 3600) * fs: (right_point + 3600) * fs].__contains__(1):
                    speech += 1
                else:
                    silence += 1

    return negative_sample, speech, silence


def generate_all_negative_sample(positive_sample, successful_y, start_y, duration,
                                 ratio,
                                 fs=100):
    negative_sample = []
    number_of_negative_sample = len(positive_sample) * ratio

    while len(negative_sample) < number_of_negative_sample:
        random_point = rand.randint(0, 600)
        # check left side of random point
        left_point = random_point - duration

        if left_point >= 0:
            if not successful_y[(left_point + 3600) * fs:(random_point + 3600) * fs].__contains__(
                    1) and not start_y[
                               (left_point + 3600) * fs:(random_point + 3600) * fs].__contains__(1):
                negative_sample.append(tuple([left_point + 3600, random_point + 3600]))

        right_point = random_point + duration

        if right_point <= 600:
            if not successful_y[(random_point + 3600) * fs: (right_point + 3600) * fs].__contains__(
                    1) and not start_y[
                               (random_point + 3600) * fs: (right_point + 3600) * fs].__contains__(1):
                negative_sample.append(tuple([random_point + 3600, right_point + 3600]))

    return negative_sample


def make_vad(df: pd.DataFrame, pid, size=9900, fs=100):
    ''' len is in seconds
    '''
    time_window = []

    vad = np.zeros((size * fs))
    '''
    loop the rttm files 
    
    '''
    for idx, row in df.iterrows():
        spk = int(row['speaker'].split('_')[1])
        if pid in main_speakers and spk not in main_speakers[pid]:
            continue

        print("ini is :  ", row['ini'])
        ini = round(row['ini'] * fs)
        end = round((row['ini'] + row['dur']) * fs)

        # 2 seconds
        time_ini = round((row['ini'] - 2) * fs)
        time_end = round(row['ini'] * fs)
        time_window.append(tuple([time_ini, time_end]))
        vad[ini:end] = 1

    return vad, time_window


def store_vad(df: pd.DataFrame, pid, fname, size=9900, fs=100):
    vad, time_list = make_vad(df, pid, size=size, fs=fs)

    # write csv
    with open(fname + '.csv', "w") as f_f:
        csv_writer = csv.writer(f_f)
        for mytuple in time_list:
            csv_writer.writerow(mytuple)

    # write .vad file
    np.savetxt(fname + '.vad', vad, fmt='%d')
    # write .wav file
    wavwrite(fname + '.wav', fs, vad)


def load_filter_vad(vad_path_filter, pid_list_filter):
    vad = {}
    for i in range(0, len(pid_list_filter)):
        temp = wavfile.read(vad_path_filter + str(pid_list_filter[i]) + ".wav")
        vad[pid_list_filter[i]] = temp[1]
        print("in load filter vad")
    if len(vad) == 0:
        print('load_vad called but nothing loaded.')

    return vad


# make vad files
from pathlib import Path

def generate_successful_train_csv(windowSize, ratio, vad_dict):
    for k in range(0, len(pid_list)):
        print("pid : ", pid_list[k])
        training_sample, training_label = generate_training_sample(vad_dict[pid_list[k]], windowSize, ratio)

        with open('./successful_train_samples/' + str(windowSize) + "s/" + "_" + str(
                pid_list[k]) + '.csv', "w") as f:
            csv_writer_new = csv.writer(f)
            for time_tuple in training_sample:
                csv_writer_new.writerow(time_tuple)
        f.close()

        outfile = open('./successful_train_ground_truth/' + str(windowSize) + "s/" + str(pid_list[k]) + '.csv', "w",
                       newline='')
        out = csv.writer(outfile)
        out.writerows(map(lambda x: [x], training_label))
        outfile.close()


def generate_successful_test_csv(windowSize, ratio, number_of_experiment, vad_dict):
    global train_speech_total
    global train_silence_total
    global test_speech_total
    global test_silence_total
    for sample_index in range(0, number_of_experiment):
        for k in range(0, len(pid_list)):

            print("index: ", sample_index, "  pid: ", k)

            test_samples, successful_test_label = generate_successful_test_sample(vad_dict[pid_list[k]], windowSize, ratio)

            # write csv file for successful test dataset
            with open('./successful_test_samples/'+ str(windowSize) + "s/" + str(sample_index) + "_" + str(pid_list[k]) + '.csv', "w") as ff:
                test_csv_writer_new = csv.writer(ff)
                for test_time_tuple in test_samples:
                    test_csv_writer_new.writerow(test_time_tuple)
            ff.close()

            # write csv file of ground truth label for successful intention case
            if sample_index == 0:
                outfile = open('./successful_test_ground_truth/' + str(windowSize) +"s/" + str(pid_list[k]) + '.csv', "w", newline='')
                out = csv.writer(outfile)
                out.writerows(map(lambda x: [x], successful_test_label))
                outfile.close()

def generate_selected_unsuccessful_csv(category: str, windowSize, ratio, number_of_experiment, vad_dict):
    """
    generate positive and negative testing samples for unsuccessful intention case.

    :param category: start, continue or both
    :param windowSize: 1-4
    :param ratio: positive samples/negative samples
    :param number_of_experiment: how many experiment run
    :return:
    """
    global total_speech
    global total_silence
    unsuccessful_positive_sample = []
    unsuccessful_label = []

    for sample_index in range(0, number_of_experiment):
        for k in range(0, len(pid_list)):

            print("index: ", sample_index, "  pid: ", k)

            all_test_sample, successful_test_label = generate_successful_test_sample(
                vad_dict[pid_list[k]], windowSize, ratio)

            # three different categories labeled unsuccessful intention samples
            start_list, continue_list, maybe_list, all_list = preprocess_unsuccessful_intention(unsuccessful_intention)

            # generate positive samples
            if category == "start":
                # generate positive samples for selected category unsuccessful intention case.
                unsuccessful_label, unsuccessful_positive_sample = generate_unsuccessful_intention_samples(pid_list[k],
                                                                                                           start_list,
                                                                                                           windowSize)

            elif category == "continue":
                unsuccessful_label, unsuccessful_positive_sample = generate_unsuccessful_intention_samples(pid_list[k],
                                                                                                           continue_list,
                                                                                                           windowSize)

            elif category == "all_unsuccessful":
                print("making all_unsuccessful")
                unsuccessful_label, unsuccessful_positive_sample = generate_unsuccessful_intention_samples(pid_list[k],
                                                                                                           all_list,
                                                                                                           windowSize)

            # generate negative samples

            unsuccessful_negative_sample, speech, silence = generate_negative_unsuccessful_sample(vad_dict[pid_list[k]],
                                                                                                  unsuccessful_positive_sample,
                                                                                                  successful_test_label,
                                                                                                  unsuccessful_label,
                                                                                                  windowSize,
                                                                                                  ratio)

            total_speech += speech
            total_silence += silence
            print(len(unsuccessful_positive_sample), "  ", len(unsuccessful_negative_sample))
            # all_unsuccessful samples of unsuccessful intention start case.
            unsuccessful_samples = unsuccessful_positive_sample + unsuccessful_negative_sample
            # shuffle positive and negative unsuccessful samples
            np.random.shuffle(unsuccessful_samples)

            # write test data and ground truth
            if start_pid.__contains__(pid_list[k]):

                with open('./unsuccessful_intention_test_sample/' + category + '/' + str(windowSize) + "s/" + str(sample_index) + "_" + str(
                        pid_list[k]) + '.csv',
                          "w") as fff:
                    unsuccessful_csv_writer_new = csv.writer(fff)
                    for unsuccessful_tuple in unsuccessful_samples:
                        unsuccessful_csv_writer_new.writerow(unsuccessful_tuple)

                # write ground truth label for unsuccessful start case.
                if sample_index == 0:  # only write once
                    outfile_unsuccessful_intention_label = open(
                        './unsuccessful_intention_test_label/' + category + '/' + str(windowSize) + "s/" + str(pid_list[k]) + '.csv', "w",
                        newline='')
                    out_unsuccessful_label = csv.writer(outfile_unsuccessful_intention_label)
                    out_unsuccessful_label.writerows(map(lambda x: [x], unsuccessful_label))
                    outfile_unsuccessful_intention_label.close()


def genrate_all(windowSize, ratio, number_of_experiment, vad_dict):
    for sample_index in range(0, number_of_experiment):

        for k in range(0, len(pid_list)):

            print(sample_index ,  " : ", pid_list[k])

            successful_test_sample, successful_test_label = generate_successful_test_sample(vad_dict[pid_list[k]], windowSize, ratio)

            # three different categories labeled unsuccessful intention samples
            start_list, continue_list, maybe_list, all_list = preprocess_unsuccessful_intention(unsuccessful_intention)

            unsuccessful_label, unsuccessful_positive_sample = generate_unsuccessful_intention_samples(pid_list[k],
                                                                                                       all_list,
                                                                                                       windowSize)

            unsuccessful_negative_sample, speech, silence = generate_negative_unsuccessful_sample(vad_dict[pid_list[k]],
                                                                                                  unsuccessful_positive_sample,
                                                                                                  successful_test_label,
                                                                                                  unsuccessful_label,
                                                                                                  windowSize,
                                                                                                  ratio)



            all_test_samples = successful_test_sample + unsuccessful_positive_sample + unsuccessful_negative_sample
            print(type(all_test_samples))
            np.random.shuffle(all_test_samples)

            # For unsuccessful case, ground truth for both start and continue samples.
            temp = list(map(operator.add, successful_test_label, unsuccessful_label))
            all_label = [1 if x > 0 else 0 for x in temp]
            print()


            # For unsuccessful case, write all_unsuccessful positive and negative samples of both start and continue case into
            # csv file
            with open('./all_sample/' + str(windowSize) + "s/" + str(sample_index) + "_" + str(pid_list[k]) + '.csv',
                      "w") as fff_all_sample:
                all_sample_csv_writer_new = csv.writer(fff_all_sample)
                for all_tuple in all_test_samples:
                    all_sample_csv_writer_new.writerow(all_tuple)

            # write ground truth label for all_unsuccessful unsuccessful start and continue case.
            if sample_index == 0:
                outfile_all_label = open(
                    './all_label/' + str(windowSize) + "s/" + str(pid_list[k]) + '.csv', "w", newline='')
                out_all_label = csv.writer(outfile_all_label)
                out_all_label.writerows(map(lambda x: [x], all_label))
                outfile_all_label.close()



def main(typeNum, windowSize, ratio, vad_dict, numOfExperiment=None, unsuccessful_category=None):
    if typeNum == 2:
        generate_successful_test_csv(windowSize, ratio, numOfExperiment, vad_dict)

    elif typeNum == 3 or typeNum == 4 or typeNum == 5:
        generate_selected_unsuccessful_csv(unsuccessful_category, windowSize, ratio, numOfExperiment, vad_dict)

    elif typeNum == 1:
        genrate_all(windowSize, ratio, numOfExperiment, vad_dict)

    elif typeNum == 0:
        generate_successful_train_csv(windowSize, ratio, vad_dict)

"""
main function
"""
if __name__ == '__main__':
    vad_dict = load_filter_vad(vad_path, pid_list)

    """
    main function
    
    the 1st parameter: 1 indicates generating successful intention case.
                         2 indicates generating unsuccessful intention case. which will need one furhter parameter.
                         
    the 2nd parameter : window size
    the 3rd parameter : positve/negative samples ratio
    the 4rd parameter : how many experiment run
    the 5rd parameter (opt) : if the 1st parameter is 2 (unsuccessful case), 
                    then need to indicate (start, continue, or all_unsuccessful) category.
    """

    for window_size in range(1,5):
        # generate training samples.
        main(0, window_size, 20, vad_dict)

        # experiment 1  done
        # generate training dataset
        main(1, window_size, 20, vad_dict, 100)


        # experiment 2 (only changes the timewindow (2nd para) and ratio (3rd para))
        # generate testing dataset for successful intention case.
        main(2, window_size, 20, vad_dict, 100)

        # experiment 3 done
        main(3, window_size, 20, vad_dict, 100, 'all_unsuccessful') # start/continue/all_unsuccessful

        # experiment 4 done
        main(4, window_size, 20, vad_dict, 100, 'start')

        # experiment 5  done
        main(5, window_size, 20, vad_dict, 100, 'continue')


