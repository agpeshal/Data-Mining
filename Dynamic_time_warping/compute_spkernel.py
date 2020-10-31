from __future__ import division, print_function
import numpy as np
import scipy.io
import argparse
import shortest_path_kernel as spk
import os


parser = argparse.ArgumentParser(description="Compute SP Kernel")
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

file_name = "MUTAG.mat"
file_path = "{}/{}".format(args.datadir, file_name)

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

out_file = "graphs_output.txt"
out_path = "{}/{}".format(args.outdir, out_file)
f_out = open(out_path, "w")

f_out.write("Pair of classes\tSPKernel\n")

mat = scipy.io.loadmat(file_path)
label = np.reshape(mat["lmutag"], (len(mat["lmutag"])))
data = np.reshape(mat["MUTAG"]["am"], len(label))


mut = list()
non_mut = list()

for i in range(len(label)):
    if label[i] == 1:
        mut.append(spk.floyd_warshall(data[i]))
    else:
        non_mut.append(spk.floyd_warshall(data[i]))

data_dic = {"mutagenic": mut, "non-mutagneic": non_mut}

types = list(data_dic.keys())

for idx1 in range(len(types)):
    for idx2 in range(idx1, len(types)):
        count = 0
        sim = 0
        for i in range(len(data_dic[types[idx1]])):
            for j in range(len(data_dic[types[idx2]])):
                if (i < j) | (idx1 != idx2):
                    count += 1
                    adj_mat1 = data_dic[types[idx1]][i]
                    adj_mat2 = data_dic[types[idx2]][j]
                    sim += spk.spkernel(adj_mat1, adj_mat2)

        f_out.write(
            "{}:{}\t{:.2f}\n".format(types[idx1], types[idx2], float(sim) / count)
        )
