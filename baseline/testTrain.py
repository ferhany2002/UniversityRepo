import os

import logging
import pickle
import traceback
import pkg_resources

import torch
torch.set_float32_matmul_precision('high')
import numpy as np

from sklearn.model_selection import KFold, GroupKFold
from IPython.display import clear_output

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
# set the cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from data_loading.dataset import FatherDataset, FatherDatasetSubset
from data_loading.extractors import AccelExtractor

from train import System, train, test

import random

'''
do_train : 
'''

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def do_cross_validation(do_train, ds, last_test_ds, input_modalities, seed, prefix=None, deterministic=False):
    # cv_splits : [array, array, array, ...]
    # split data into 3 sets
    # group number = pid number

    last_test_set_idx = np.arange(len(last_test_ds.examples))


    cv_splits = list(GroupKFold(n_splits=3).split(range(len(ds)), groups=ds.get_groups()))

    all_results = []
    roc_list = None
    cross_validation_roc = []

    for f, (train_idx, test_idx) in enumerate(cv_splits):
        print("f :", f, "   train_idx : ", train_idx, " ", type(train_idx), "  test_idx : ", len(test_idx), "  ",
              type(test_idx))
        if f == 0 or f == 1:
            continue


        train_ds = FatherDatasetSubset(ds, train_idx, eval=False)
        test_ds = FatherDatasetSubset(ds, test_idx, eval=True)

        # last test set
        real_test_ds = FatherDatasetSubset(last_test_ds, last_test_set_idx, eval=True)

        weights_path = os.path.join(
            'weights',
            f'I{"-".join(input_modalities)}_fold{f}.ckpt'
        )

        seed_everything(seed + f + 734890573)
        if do_train:
            trainer, roc_list = train(f, train_ds, test_ds, input_modalities,
                                                                        prefix=prefix + f'_fold{f}' if prefix else None,
                                                                        eval_every_epoch=True,
                                                                        deterministic=deterministic,
                                                                        weights_path=weights_path)
            model = trainer.model

            torch.save(model.state_dict(), "model_SaveContinue1s200.pt")

        else:

            model = System('accel', 'classification')
            model.load_state_dict(torch.load("model_SaveContinue1s200.pt"))

        # select the best model

        # ensures that the testing is reproducible regardless of training
        seed_everything(seed + f + 2980374334)
        fold_outputs = test(f, model, real_test_ds, prefix=prefix + f'_fold{f}' if prefix else None, )
        all_results.append(fold_outputs)

        # reuslts for different cross validation set
        if do_train:
            roc_list.append(f)
            cross_validation_roc.append(roc_list)


        clear_output(wait=False)

    # save the best model after cross validation

    outputs = [r['proba'].numpy() for r in all_results]
    indices = [r['index'].numpy() for r in all_results]
    metrics = [r['metric'] for r in all_results]
    precision = [r['precision'] for r in all_results]
    recall = [r['recall'] for r in all_results]

    # return metrics, outputs, indices, precision, recall, roc_list
    return f, metrics, outputs, indices, precision, recall, cross_validation_roc


def do_run(examples, test_examples, input_modalities,
           do_train=True, deterministic=True, prefix=''):
    ''' train = True will train the models, and requires
            model_label_modality = test_label_modality
        train = False will load weights to test the models and does not require
            model_label_modality = test_label_modality
    '''
    print(f'Using {len(examples)} examples')
    print(f'Using {len(test_examples)} test examples')
    # create the feature datasets
    extractors = {}

    if 'accel' in input_modalities:
        # get accel data
        accel_ds_path = '../data/subj_accel_interp.pkl'
        extractors['accel'] = AccelExtractor(accel_ds_path)

    # extract data based on features selected
    ds = FatherDataset(examples, extractors)
    test_ds = FatherDataset(test_examples, extractors)

    seed = 22

    f_fold, metrics, probas, indices, precision, recall, cross_validation_roc = do_cross_validation(
            do_train,
            ds,
            test_ds,
            input_modalities=input_modalities,
            deterministic=deterministic,
            seed=seed,
            prefix=f'{prefix}I{"-".join(input_modalities)}')

    torch.cuda.empty_cache()

    return {
               'f_fold': f_fold,
               'metrics': metrics,
               'probas': probas,
               'indices': indices,
               'seed': seed,
               'precision': precision,
               'recall': recall
           }, cross_validation_roc


def get_table(index_i, Num, windowSize, do_train=True, deterministic=True):

    train_examples = pickle.load(open("../data/train_pkl/" + str(windowSize) + "s/" + "_INTS_train.pkl", 'rb'))
    test_examples = None

    if Num == 2:
        print("in experiment 2")
        test_examples = pickle.load(open("../data/successful_test_pkl/" + str(windowSize) + "s/" + str(index_i) +"_INTS_test.pkl", 'rb'))

    elif Num == 1:
        print("in experiment 1")
        test_examples = pickle.load(open("../data/all_test_pkl/" + str(windowSize) + "s/" + str(index_i) +"_INTS_test.pkl", 'rb'))

    elif Num == 3:
        print("in experiment 3")
        test_examples = pickle.load(
            open("../data/unsuccessful_test_pkl/all_unsuccessful/" + str(windowSize) + "s/" + str(index_i) + "_INTS_test.pkl", 'rb'))

    elif Num == 4:
        print("in experiment 4")
        test_examples = pickle.load(
            open("../data/unsuccessful_test_pkl/start/" + str(windowSize) + "s/" + str(index_i) + "_INTS_test.pkl", 'rb'))

    elif Num == 5:
        print("in experiment 5")
        test_examples = pickle.load(
            open("../data/unsuccessful_test_pkl/continue/" + str(windowSize) + "s/" + str(index_i) + "_INTS_test.pkl", 'rb'))

    elif Num == 0:
        test_examples = pickle.load(
            open("../data/successful_test_pkl/" + str(windowSize) + "s/" + "0_INTS_test.pkl", 'rb'))

    all_input_modalities = [
        # ('video',),
        # ('pose',),
        ('accel',),
    ]

    res = {}
    cross_validation_roc = []
    '''
    examples: 输入的数据
    '''
    for input_modalities in all_input_modalities:
        run_results, cross_validation_roc = do_run(
            train_examples,
            test_examples,
            input_modalities,
            do_train=do_train,
            deterministic=deterministic)

        res['-'.join(input_modalities)] = run_results
    return res, cross_validation_roc  # res

def main(train, experiment_num, windowSize, numberOfExperiment):
    try:
        if train:
            res, cross_validation_roc = get_table(0, 0, windowSize, do_train=True, deterministic=False)

            print(res)
            print('\n\n\n\n')

            print(cross_validation_roc)

        else:
            # output result in txt file.
            with open('./result/' + "experiment" + str(experiment_num) + "/" + str(windowSize) + "s/" + "performance_new_temp_1.txt", 'w') as f:
                metric_list = []
                precision_list = []
                recall_list = []

                for index_q in range(0, numberOfExperiment):
                    print("experiment : ", experiment_num, "  index : ", index_q)
                    res, cross_validation_roc = get_table(index_q, experiment_num, windowSize, do_train=False, deterministic=False)
                    print(res)
                    for ks, vs in res.items():
                        for k, v in vs.items():
                            if k == "metrics":
                                metric_list.append(v[0])
                            if k == "precision":
                                precision_list.append(v[0])
                            if k == "recall":
                                recall_list.append(v[0])
                f.write(str(metric_list) + '\n')
                f.write(str(np.mean(metric_list)) + "  " + str(np.std(metric_list)) + '\n')
                f.write(str(precision_list) + '\n')
                f.write(str(np.mean(precision_list)) + "  " + str(np.std(precision_list)) + '\n')
                f.write(str(recall_list) + '\n')
                f.write(str(np.mean(recall_list)) + "  " + str(np.std(recall_list)))
            f.close()

    except Exception:
        print(traceback.format_exc())


if __name__ == '__main__':
    #train model
    #main(True, experiment_num=0, windowSize=1, numberOfExperiment=100)

    # experiment 1 done
    #main(True, experiment_num=0, windowSize=2, numberOfExperiment=100)

    # experiment 2 done
    #main(True, experiment_num=2, windowSize=2, numberOfExperiment=100)

    # experiment 3
    #main(True, experiment_num=3, windowSize=3, numberOfExperiment=100)

    # experiment 4
    main(False, experiment_num=5, windowSize=1, numberOfExperiment=25)

    # experiment 5
    #main(False, experiment_num=5, windowSize=4, numberOfExperiment=100)

