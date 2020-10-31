import numpy as np
import re
import argparse


def GXY(a, b):
    a = np.asarray(list(a))
    b = np.asarray(list(b))

    # We want both the 3-mers to look like GXY with X or Y not being G
    # np.char.find() returns 0 for presence and -1 for absence
    if (np.char.find("G", a) + np.char.find("G", b) == [0, -2, -2]).all():
        return sum(a == b)
    else:
        return 0


def kmer(X, Y, k):

    x_kmer = list()
    y_kmer = list()

    # re.findall('...', str) returns all substrings of length 3
    # but without any overlap
    for i in range(k):
        x_kmer.extend(re.findall("...", X[i:]))
        y_kmer.extend(re.findall("...", Y[i:]))

    return x_kmer, y_kmer


def kernel_kmer(X, Y, k=3):

    """
    Calculates the kernel between X and Y using the GXY function
    """
    x_kmer, y_kmer = kmer(X, Y, k)

    sim = 0
    for a in x_kmer:
        for b in y_kmer:
            sim += GXY(a, b)

    return sim


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Kernel between two sequences")

    parser.add_argument("--seq1", required=True, help="First sequence")
    parser.add_argument("--seq2", required=True, help="Second sequence")

    args = parser.parse_args()

    print("Kernel between them {}".format(kernel_kmer(args.seq1, args.seq2)))
