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

def strip_lowercase_msa(filepath):
   """
  stripping the lowercase in a3m format (the deleted residues) so that all msa have the same length
  """
  a3m_msa = parse_msa_with_a3m(filepath)
  msa_seqs = []
  for seq in a3m_seqs:
    none_lowercase_letters = []
    for letter in seq:
      if letter.islower() == False:
        none_lowercase_list.append(letter)
    msa_seqs.append(''.join(none_lowercase_letters))
  return msa_seqs  