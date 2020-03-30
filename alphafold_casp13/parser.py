import numpy as np


def parse_msa_with_a3m(filepath):
    msa = []
    with open(filepath) as fp:
       line = fp.readline()
       cnt = 1
       while line:
           # skip information lines
           if cnt >3  and cnt % 2 != 0:
              msa.append(line.strip())
           line = fp.readline()
           cnt += 1
    return msa

