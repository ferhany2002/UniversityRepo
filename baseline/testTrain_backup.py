import os
import sys
import logging
import pickle
import traceback

import torch
import lightning_lite
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from IPython.display import clear_output
# import pytorch_lightning as pl
import lightning_lite as pl

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
# set the cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from data_loading.dataset import FatherDataset, FatherDatasetSubset
from data_loading.extractors import AccelExtractor
from constants import (
    processed_data_path,
    processed_accel_path,
    processed_videos_path,
    examples_path, dataset_path)
from train import System, train, test

import multiprocessing
from multiprocessing import Process
import threading

import matplotlib.pyplot as plt

'''
do_train : 
'''
from model import SegmentationFusionModel


def do_cross_validation(do_train, ds, last_test_ds, input_modalities, seed, prefix=None, deterministic=False):
    # cv_splits : [array, array, array, ...]
    # split data into 3 sets
    # group number = pid number

    last_test_set_idx = np.arange(len(last_test_ds.examples))

    # print(" len of test ds : ", len(test_ds), " type : ", type(test_ds))
    cv_splits = list(GroupKFold(n_splits=3).split(range(len(ds)), groups=ds.get_groups()))
    # for i in range(0, len(cv_splits)):
    #     print("type : ", type(cv_splits[i]),"  : ", len(cv_splits[i][0]), "  # ", len(cv_splits[i][1]))
    all_results = []
    roc_list = None
    cross_validation_roc = []

    for f, (train_idx, test_idx) in enumerate(cv_splits):
        print("f :", f, "   train_idx : ", train_idx, " ", type(train_idx), "  test_idx : ", len(test_idx), "  ",
              type(test_idx))
        if f == 0 or f == 1:
            continue

        # load feature caches for fold f
        #
        # ###########################  make weight :
        # train_all_label = ds.get_all_labels()
        # temp = np.take(train_all_label,train_idx)
        # temp_tensor = torch.from_numpy(temp)
        # class_sample_count = torch.tensor(
        #     [(temp_tensor == t).sum() for t in torch.unique(temp_tensor, sorted=True)])
        # weight = 1. / class_sample_count.float()
        # train_samples_weight = torch.tensor([weight[int(t)] for t in temp_tensor])
        # print("train : ", weight, train_samples_weight)
        #
        #
        # ########  val weight
        # train_all_label = ds.get_all_labels()
        # temp_val = np.take(train_all_label,test_idx)
        # temp_val_tensor = torch.from_numpy(temp_val)
        # class_val_sample_count = torch.tensor(
        #     [(temp_val_tensor == t).sum() for t in torch.unique(temp_val_tensor, sorted=True)])
        # val_weight = 1. / class_val_sample_count.float()
        # val_samples_weight = torch.tensor([val_weight[int(t)] for t in temp_val_tensor])
        # print("val : ", val_weight, val_samples_weight)

        ################################################

        train_ds = FatherDatasetSubset(ds, train_idx, eval=False)
        test_ds = FatherDatasetSubset(ds, test_idx, eval=True)

        # last test set
        real_test_ds = FatherDatasetSubset(last_test_ds, last_test_set_idx, eval=True)

        weights_path = os.path.join(
            'weights',
            f'I{"-".join(input_modalities)}_fold{f}.ckpt'
        )

        pl.utilities.seed.seed_everything(seed + f + 734890573)
        if do_train:
            trainer, roc_list = train(f, train_ds, test_ds, input_modalities,
                                                                        prefix=prefix + f'_fold{f}' if prefix else None,
                                                                        eval_every_epoch=True,
                                                                        deterministic=deterministic,
                                                                        weights_path=weights_path)
            model = trainer.model

            torch.save(model.state_dict(), "!!!.pt")

            # if best_model is None:
            #     best_model = model
            #     best_model_performance = roc_list
            #
            # else:
            #     if roc_list[-1] > best_model_performance[-1]:
            #         best_model = model
            #         best_model_performance = roc_list
        else:

            # model = System.load_from_checkpoint(checkpoint_path=weights_path)
            model = System('accel', 'classification')
            model.load_state_dict(torch.load("model_version_4_20_10.pt"))

        # select the best model

        # ensures that the testing is reproducible regardless of training
        pl.utilities.seed.seed_everything(seed + f + 2980374334)
        fold_outputs = test(f, model, real_test_ds, prefix=prefix + f'_fold{f}' if prefix else None, )
        all_results.append(fold_outputs)

        # reuslts for different cross validation set
        if do_train:
            roc_list.append(f)
            cross_validation_roc.append(roc_list)
            # torch.save(best_model.state_dict(), "best_model.pt")

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
        # accel_ds_path = os.path.join(processed_accel_path, 'subj_accel_interp.pkl')
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
    # examples = pickle.load(open(examples_path, 'rb'))
    # data set
    train_examples = pickle.load(open("../data/train_pkl/" + str(windowSize) + "s/" + "_INTS_train.pkl", 'rb'))
    test_examples = None

    if Num == 2:
        test_examples = pickle.load(open("../data/successful_test_pkl/" + str(windowSize) + "s/" + str(index_i) +"_INTS_test.pkl", 'rb'))

    elif Num == 1:
        test_examples = pickle.load(open("../data/all_test_pkl/" + str(windowSize) + "s/" + str(index_i) +"_INTS_test.pkl", 'rb'))

    elif Num == 3:
        test_examples = pickle.load(
            open("../data/unsuccessful_test/all_unsuccessful/" + str(windowSize) + "s/" + str(index_i) + "_INTS_test.pkl", 'rb'))

    elif Num == 4:
        test_examples = pickle.load(
            open("../data/unsuccessful_test/start/" + str(windowSize) + "s/" + str(index_i) + "_INTS_test.pkl", 'rb'))

    elif Num == 5:
        test_examples = pickle.load(
            open("../data/unsuccessful_test/continue/" + str(windowSize) + "s/" + str(index_i) + "_INTS_test.pkl", 'rb'))

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

def main(train, Num, windowSize, numberOfExperiment=100):
    try:
        if train:
            res, cross_validation_roc = get_table(0, 0, windowSize, do_train=True, deterministic=False)

            print(res)
            print('\n\n\n\n')

            print(cross_validation_roc)

        else:
            # output result in txt file.
            with open('./result/' + "experiment" + str(Num) + "/"+ str(windowSize) + "s/" + "performance.txt", 'w') as f:
                metric_list = []
                precision_list = []
                recall_list = []

                for index_q in range(0, 100):
                    print("index : ", index_q)
                    res, cross_validation_roc = get_table(index_q, Num, do_train=False, deterministic=False)
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
    main(True, 0, 1, 100)
