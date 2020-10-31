#!/usr/bin/env python
# coding: utf-8

from __future__ import division
import numpy as np
import argparse
import os


def dist(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


parser = argparse.ArgumentParser(description="K Nearest Neighbour")

parser.add_argument("--traindir", required=True, help="Path to training data directory")
parser.add_argument("--testdir", required=True, help="Path to test data directory")
parser.add_argument("--outdir", required=True, help="Path to the output directory")
parser.add_argument("--mink", required=True, help="Minimum value of k")
parser.add_argument("--maxk", required=True, help="Maximum value of k")

args = parser.parse_args()

inp_file_name = "matrix_mirna_input.txt"
class_file = "phenotype.txt"

train_inp_path = os.path.join(args.traindir, inp_file_name)
train_class_path = os.path.join(args.traindir, class_file)

test_inp_path = os.path.join(args.testdir, inp_file_name)
test_class_path = os.path.join(args.testdir, class_file)

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

out_file_path = os.path.join(args.outdir, "output_knn.txt")


with open(train_inp_path, "r") as f_tr:

    next(f_tr)
    dict_mRNA_train = {}

    for line in f_tr:
        parts = line.rstrip().split("\t")

        dict_mRNA_train[parts[0]] = np.asarray(list(map(float, parts[1:])))


with open(test_inp_path, "r") as f_tst:

    next(f_tst)
    dict_mRNA_test = {}

    for line in f_tst:
        parts = line.rstrip().split("\t")

        dict_mRNA_test[parts[0]] = np.asarray(list(map(float, parts[1:])))


with open(train_class_path, "r") as f_map:
    next(f_map)
    dict_class_train = {}

    for line in f_map:
        parts = line.rstrip().split("\t")

        dict_class_train[parts[0]] = parts[1]

f_map.close()


with open(test_class_path, "r") as f_map:
    next(f_map)
    dict_class_test = {}

    for line in f_map:
        parts = line.rstrip().split("\t")

        dict_class_test[parts[0]] = parts[1]


# Creating output file

f_out = open(out_file_path, "w")

header = "\t".join(["Value of k", "Accuracy", "Precision", "Recall"])
f_out.write("{}\n".format(header))


for k in range(int(args.mink), int(args.maxk) + 1):

    dict_class_pred = {}

    for p_id, mRNA in dict_mRNA_test.items():
        d = {}

        for p_id_tr, mRNA_tr in dict_mRNA_train.items():
            d[p_id_tr] = dist(mRNA, mRNA_tr)  # Store distance from each patient ID

        # sorted Pateint IDs based on distances
        id_sorted = sorted(d, key=d.__getitem__)

        # stores the label values of patients in increasing order of distances
        label = []
        # If the k is even, just ignore the kth largest distance to avoid conflict,
        # which always gives the same result as for k-1
        if k % 2 == 0:
            t = k - 1
        else:
            t = k

        for id_srt in id_sorted[:t]:
            label.append(dict_class_train[id_srt])

        # return the label with the max frequency count
        dict_class_pred[p_id] = max(label, key=label.count)

    true_class = list(dict_class_test.values())
    pred_class = list(dict_class_pred.values())
    pos = [x == "+" for x in pred_class]
    neg = [x == "-" for x in pred_class]
    true = [x == "+" for x in true_class]
    false = [x == "-" for x in true_class]

    tp = np.sum(np.logical_and(true, pos))
    fp = np.sum(np.logical_and(false, pos))
    fn = np.sum(np.logical_and(true, neg))
    tn = np.sum(np.logical_and(false, neg))

    accuracy = (tp + tn) / (tn + fp + fn + tp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    vec_metric = [accuracy, precision, recall]

    str_metric = "\t".join("{0:.2f}".format(x) for x in vec_metric)

    f_out.write("{}\t{}\n".format(k, str_metric))
