from __future__ import division, print_function
import numpy as np
from math import inf
import dynamic_time_warping as dtw
import argparse
import os

# if __name__ == '__main__':
# 	main()

parser = argparse.ArgumentParser(description="Compute average distances between groups")
parser.add_argument(
    "--datadir",
    required=True,
    help="the path to the directory where the input file is stored",
)
parser.add_argument(
    "--outdir",
    required=True,
    help="the path to the directory where the output file will be stored",
)
args = parser.parse_args()

file_name = "ECG200_TRAIN.txt"
file_path = "{}/{}".format(args.datadir, file_name)

series = list()
label = list()
with open(file_path, "r") as file:
    for line in file.readlines():
        l = line.replace("\n", "")
        vec = l.split(",")
        vec = list(map(float, vec))
        label.append(int(vec[0]))
        series.append(np.asarray(vec[1:]))

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

series_norm = list()
series_abnorm = list()

n = len(series)

for i in range(n):
    if label[i] == 1:
        series_norm.append(series[i])
    else:
        series_abnorm.append(series[i])

data_dic = {"abnormal": series_abnorm, "normal": series_norm}
ts_type = list(data_dic.keys())

constraints = [0, 10, 25, inf]

out_file = "timeseries_output.txt"
out_path = "{}/{}".format(args.outdir, out_file)
f_out = open(out_path, "w")
f_out.write(
    "Pair of classes\tManhattan\tDTW, w = 0\tDTW, w = 10\tDTW, w = 25\tDTW, w = inf\n"
)

for idx1 in range(len(ts_type)):
    for idx2 in range(idx1, len(ts_type)):

        t1 = data_dic[ts_type[idx1]]
        t2 = data_dic[ts_type[idx2]]

        manh_avg = 0
        dtw_avg = np.zeros(len(constraints))
        count = 0

        for i in range(len(t1)):
            for j in range(len(t2)):
                if (i < j) | (idx1 != idx2):
                    count += 1
                    manh_avg += dtw.dist(t1[i], t2[j])
                    for k, w in enumerate(constraints):
                        dtw_avg[k] += dtw.constrained_dtw(t1[i], t2[j], w)

        sim_vec = np.insert(dtw_avg / count, 0, manh_avg / count)
        str_vec = "\t".join("{0:.2f}".format(x) for x in sim_vec)

        f_out.write("{}:{}\t{}\n".format(ts_type[idx1], ts_type[idx2], str_vec))
