
import pickle

import utils
from sklearn.metrics import accuracy_score
accel_path = "../average_vectors.pkl"

target_words = {"maar", "van", "nou",  "en", "uh", "nou", "ook", "ik", "juist", "nee"}

def rule_based_classification(words):
    """
    This function takes in a list of words and a set of target words. It returns 1
    if any of the words are in the set of target words, and 0 otherwise. ONLY WORKS FOR WINDOW SIZE 1
    """
    for word in words:
        if word.lower() in target_words:
            return 1
    return 0

def test_with_rule_based_classifier(test_samples):
    #Only works for window size 1

    true_labels = []
    rule_based_predictions = []

    # Apply the rule-based classifier to each example
    for example in test_samples:
        true_label = example['vad']
        rule_based_prediction = rule_based_classification(example['words'])

        true_labels.append(true_label)
        rule_based_predictions.append(rule_based_prediction)

    # Compute the accuracy
    accuracy = accuracy_score(true_labels, rule_based_predictions)

    print("ACCURACY = ")
    print(accuracy)

    return accuracy

def make_all_examples(Num, windowSize, feature_fs, numberOfExperiment=None, category=None):
    accel_path = "../average_vectors.pkl"

    # successful train ground truth label
    realized_intention_label_path = "../preprocess/audio/successful_train_ground_truth/"


    # unsuccessful intention start/continue case ground truth. (which one depends on selection)
    unrealized_intention_path = "../preprocess/audio/unsuccessful_intention_test_label/"

    # Both start and continue unsuccessful intention ground truth label.
    all_intention_label_path = "../preprocess/audio/all_label/"

    start_pid = [2, 3, 4, 5, 7, 10, 11, 17, 22, 23, 27, 34, 35]


    if Num == 1:
        temp = all_intention_label_path + str(windowSize) + "s/"
        maker = utils.Maker(accel_path=accel_path, all_sample_path=temp)
        for index in range(0, numberOfExperiment):
            print("num : 1 -- ", index)
            all_test_samples = maker.make_all_examples(index, windowSize, feature_fs)
            print("Number of all test samples:", len(all_test_samples))
            pickle.dump(all_test_samples, open('../data/all_test_pkl/' +  str(windowSize) + "s/" + str(index) + '_INTS_test.pkl', 'wb'))
            accuracy = test_with_rule_based_classifier(all_test_samples)

    elif Num == 2:
        temp = realized_intention_label_path + str(windowSize) + "s/"
        maker = utils.Maker(accel_path=accel_path, vad_path=temp)
        for index in range(0, numberOfExperiment):
            print("num : 2 -- ", index)

            successful_test_samples = maker.make_test_examples(index, windowSize, feature_fs)
            print("Number of successful test samples:", len(successful_test_samples))
            pickle.dump(successful_test_samples, open('../data/successful_test_pkl/' + str(windowSize) + "s/" + str(index) + '_INTS_test.pkl', 'wb'))
            accuracy = test_with_rule_based_classifier(successful_test_samples)
    elif Num == 0:
        print("Generate training pkl")
        temp = realized_intention_label_path + str(windowSize) + "s/"
        maker = utils.Maker(accel_path=accel_path, vad_path=temp)
        train_samples = maker.make_train_examples(windowSize, feature_fs)
        print("Number of train samples:", len(train_samples))
        pickle.dump(train_samples, open('../data/train_pkl/' + str(windowSize) + "s/" + '_INTS_train.pkl', 'wb'))

    elif Num == 3 or Num == 4 or Num == 5:
        temp = unrealized_intention_path + str(category) + "/" + str(windowSize) + "s/"
        maker = utils.Maker(accel_path=accel_path, unsuccessful_vad_path=temp)
        for index in range(0, numberOfExperiment):
            print("num : 3-4-5 -- ", index, "  ", str(category))
            all_test_samples = maker.make_unsuccessful_examples(start_pid, index, windowSize, feature_fs, category)
            print("Number of unsuccessful test samples:", len(all_test_samples))
            pickle.dump(all_test_samples, open('../data/unsuccessful_test_pkl/' + str(category) + "/" + str(windowSize) + "s/" + str(index) + '_INTS_test.pkl', 'wb'))
            accuracy = test_with_rule_based_classifier(all_test_samples)


def main(Num, numberOfExperiment, category=None):
    if Num == 1:
        make_all_examples(Num, numberOfExperiment)
    elif Num == 2:
        make_all_examples(Num, numberOfExperiment)
    elif Num == 3 or  numberOfExperiment == 4 or  numberOfExperiment == 5:
        make_all_examples(Num, numberOfExperiment, category)
    elif Num == 0:
        make_all_examples(0, 1)


if __name__ == '__main__':

    for window_size in range(1,2):
        # print("window size : ", window_size)
        # # experiment 0
        # make_all_examples(0, window_size, feature_fs=1)
        #
        # # experiment 1
        make_all_examples(1, window_size, feature_fs=1, numberOfExperiment=2)

        # # experiment 2  done
        #make_all_examples(2, window_size, feature_fs=1, numberOfExperiment=100)
        #
        # # experiment 3
        make_all_examples(3, window_size, feature_fs=1, numberOfExperiment=1, category='all_unsuccessful')
        # #
        # # # experiment 4
        make_all_examples(4, window_size, feature_fs=1, numberOfExperiment=1, category='start')
        # #
        # # # experiment 5
        make_all_examples(5, window_size, feature_fs=1, numberOfExperiment=1, category='continue')




