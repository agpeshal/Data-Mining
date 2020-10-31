from __future__ import division
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Naive Bayes")

parser.add_argument("--traindir", required=True, help="Training data directory")
parser.add_argument("--outdir", required=True, help="Output directory ")


args = parser.parse_args()

file_name = "tumor_info.txt"

train_file_path = os.path.join(args.traindir, file_name)

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

out_file_path = os.path.join(args.outdir, "output_summary_class_")


# Reading input data
with open(train_file_path, "r") as f_in:
    data = []
    for line in f_in:
        parts = line.rstrip().split("\t")
        # Replacing missing values with -1
        parts = list(map(lambda x: x if x != "" else -1, parts))
        # converting strings to integer
        parts = list(map(int, parts))
        data.append(parts)
f_in.close()


data_mat = np.transpose(np.asarray(data))

# Finding all types of tumors i.e. unique values in the last column

labels = np.unique(data_mat[-1])
# Converting to strings so to use them as dictionary keys
labels = [str(x) for x in labels]

# Creating tumor dictonary

tumor_map = {}
for label in labels:
    tumor_map[label] = []

for x in data:
    # last entry contains the label
    tumor_map[str(x[-1])].append(x[: len(x) - 1])


# Transposing to use each column as feature
for label in labels:
    tumor_map[label] = np.transpose(tumor_map[label])

# Caculating probability
for label, data in tumor_map.items():

    # Creating output file for each label

    f_out = open(out_file_path + label + ".txt", "w+")
    header = "\t".join(["Value", "clump", "uniformity", "marginal", "mitoses"])
    f_out.write("{}\n".format(header))

    # Multinoulli prob. vector for each feature
    param = []
    for feature in data:
        # Count of each type
        p = np.histogram(feature, range=(1, 10))[0]
        # Correction term for missing values
        non_missing = np.sum(feature != -1)
        param.append(p / non_missing)

    # Prob. for each possible value of feature
    for k in range(10):
        vec_prob = np.transpose(param)[:][k]

        str_prob = "\t".join("{0:.3f}".format(y) for y in vec_prob)

        f_out.write("{}\t{}\n".format(k + 1, str_prob))
    f_out.close()
