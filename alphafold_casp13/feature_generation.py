from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow.compat.v1 as tf
import os

import cProfile
import parser

import numpy as np
from contacts_dataset import *
from ccmpred.raw import *

tf.enable_eager_execution()

print(tf.executing_eagerly())


features = [
      "profile",
      "hhblits_profile",
      "aatype",
      "pseudo_frob",
      "pseudolikelihood",
      "deletion_probability",
      "gap_matrix",
      "pseudo_bias",
      "profile_with_prior",
      "profile_with_prior_without_gaps",
      "reweighted_profile",
      "non_gapped_profile",
      "hmm_profile",
      "num_alignments",
      "seq_length",
      "num_effective_alignments",
      "resolution",
      "sec_structure",
      "sec_structure_mask",
      "solv_surf",
      "solv_surf_mask",
      "beta_positions",
      "beta_mask",
      "domain_name",
      "chain_name",
      "resolution",
      "num_alignments",
      "superfamily",
      "profile",
      "hhblits_profile",
      "residue_index",
      "between_segment_residues",
      "mutual_information"
    ]
tf_dataset = create_tf_dataset('./casp13_data/T1006/T1006.tfrec', features)
# sequence = 'MGSSHHHHHHSSGLVPRGSHMEGKKILVTGGTGQVARPVAEALAERNEVWCLGRFGTPGVEKELNDRGITTFHWDMDDPGAAAYEGLPDDFTHVLHSAVRRGEDGDVNAAVEVNSVACGRLMTHCRGAEAFLFVSTGALYKRQTLDHAYTEDDPVDGVADWLPAYPVGKIAAEGAVRAFAQVLNLPTTIARLNIAYGPGGYGGVPMLYFKRMLAGEPIPVPKEGQNWCSLLHTDDLVAHVPRLWEAAATPATLVNWGGDEAVGITDCVRYLEELTGVRARLVPSEVTRETYRFDPTRRREITGPCRVPWREGVRRTLQALHPEHLPSESRHSAV'
np.set_printoptions(threshold=sys.maxsize)
# f= open("./T1006.txt","w+")
with open('./T1006.txt', 'a') as f:
    for feature in features:
        print(feature, file=f)
        print('\n', file=f)
        for example in tf_dataset.take(10):
            data = example[feature]
            print(data.numpy(), file=f)
        print('\n', file=f)
# f.close()

# 1. aatype
def sequence_to_onehot(sequence):
  """Maps the given sequence into a one-hot encoded matrix."""
  mapping = {aa: i for i, aa in enumerate('ARNDCQEGHILKMFPSTWYVX')}
  num_entries = max(mapping.values()) + 1
  one_hot_arr = np.zeros((len(sequence), num_entries), dtype=np.int32)

  for aa_index, aa_type in enumerate(sequence):
    aa_id = mapping[aa_type]
    one_hot_arr[aa_index, aa_id] = 1

  return one_hot_arr

#
# one_hot = sequence_to_onehot(sequence)

# print(one_hot[32:32+128][:5])
msa_file = './MSA/T1019s2/hhblits_full_6934317.a3m'
hhblits_a3m_sequences = parser.parse_msa_with_a3m(msa_file)
def deletion_probability(hhblits_a3m_sequences):
      deletion_matrix = []
      for msa_sequence in hhblits_a3m_sequences:
        deletion_vec = []
        deletion_count = 0
        for j in msa_sequence:
          if j.islower():
            deletion_count += 1
          else:
            deletion_vec.append(deletion_count)
            deletion_count = 0
        deletion_matrix.append(deletion_vec)

      deletion_matrix = np.array(deletion_matrix)
      deletion_matrix[deletion_matrix != 0] = 1.0
      deletion_probability = deletion_matrix.sum(axis=0) / len(deletion_matrix)
      deletion_probability = tf.reshape(deletion_probability, [-1, 1])
      return deletion_probability


'''
MSA = A A C D B D F J G B M A
      - - C D B D F J G B M A
      A A C D B - - J G B M A
gap_count = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]]
gap_matrix = np.matmul(gap_count.T, gap_count)
'''
def gap_matrix(hhblits_a3m_sequences):
      gap_count = []
      for msa_sequence in hhblits_a3m_sequences:
            gap_vec = []
            for j in msa_sequence:
                  if j == '-':
                        gap_vec.append(1)
                  elif j.isupper():
                        gap_vec.append(0)
            gap_count.append(gap_vec)
      gap_count = np.array(gap_count)
      # print('gap_count', gap_count)
      gap_matrix = np.matmul(gap_count.T, gap_count)
      # print('gap_matrix', gap_matrix)
      gap_matrix = gap_matrix / len(hhblits_a3m_sequences)
      gap_matrix = tf.reshape(tf.convert_to_tensor(gap_matrix), [len(gap_matrix), len(gap_matrix), 1])
      return gap_matrix


def non_gapped_profile(amino_acids):
  """Computes a profile from only amino acids and discounting gaps."""
  mapping = {aa: i for i, aa in enumerate('ARNDCQEGHILKMFPSTWYVX')}
  profile = np.zeros(21)
  for aa in amino_acids:
      aa = mapping.get(aa)
      if aa != 21:  # Ignore gaps.
            profile[aa] += 1.
  return profile / np.sum(profile)

# non_gapped_profile = non_gapped_profile(sequence)
# print('non_gapped', non_gapped_profile)

# for example in tf_dataset.take(20):
#     sequence = str(example['sequence']).split('\'')[1]
#     deletion_prob = example['deletion_probability']
#     num_alignments = example['num_alignments']
#     gap_m = example['gap_matrix']
#     gap_m = tf.reshape(gap_m, [gap_m.shape[0], gap_m.shape[1]])
#     print('sequence', sequence)
#     # print('gap_matrix', gap_m)
#
#     print('num_alignments', num_alignments[0])
    # msa_file = './MSA/T0965/{}.a3m'.format(sequence)
    # hhblits_a3m_sequences = parser.parse_msa_with_a3m(msa_file)
    # recon_data = gap_matrix(hhblits_a3m_sequences)
    # print('recon_data', recon_data)
    # print('difference', np.sum(gap_m.numpy()-recon_data))

# hmm profile
def extract_hmm_profile(hhm_file, sequence, asterisks_replace=0.0):
  """Extracts information from the hmm file and replaces asterisks."""
  profile_part = hhm_file.split('#')[-1]
  profile_part = profile_part.split('\n')
  whole_profile = [i.split() for i in profile_part]
  # This part strips away the header and the footer.
  whole_profile = whole_profile[5:-2]
  gap_profile = np.zeros((len(sequence), 10))
  aa_profile = np.zeros((len(sequence), 20))
  count_aa = 0
  count_gap = 0
  for line_values in whole_profile:
    if len(line_values) == 23:
      # The first and the last values in line_values are metadata, skip them.
      for j, t in enumerate(line_values[2:-1]):
        aa_profile[count_aa, j] = (
            2**(-float(t) / 1000.) if t != '*' else asterisks_replace)
      count_aa += 1
    elif len(line_values) == 10:
      for j, t in enumerate(line_values):
        gap_profile[count_gap, j] = (
            2**(-float(t) / 1000.) if t != '*' else asterisks_replace)
      count_gap += 1
    elif not line_values:
      pass
    else:
      raise ValueError('Wrong length of line %s hhm file. Expected 0, 10 or 23'
                       'got %d'%(line_values, len(line_values)))
  hmm_profile = np.hstack([aa_profile, gap_profile])
  assert len(hmm_profile) == len(sequence)
  return tf.convert_to_tensor(hmm_profile)

# potts model params
def pseudo_params(filepath):
      raw_output = parse_msgpack(filepath)
      x_pair = raw_output.x_pair #pseudolikelihood
      x_single = raw_output.x_single #pseudobias
      # adding entries for X and set those values to 0:
      #bias: add a coln at index 20 for the residue X
      pseudo_bias = np.insert(x_single, 20, np.zeros(x_single.shape[0]), axis=1)
      # pseudolikelihood
      h, w, c, d = x_pair.shape
      pseudolikelihood = np.insert(x_pair, 20, np.zeros((h, w, c)), axis=2)
      pseudolikelihood = np.insert(pseudolikelihood, 20, np.zeros((h, w, d+1)), axis=3)
      pseudolikelihood = pseudolikelihood.reshape((h, w, 484))
      return tf.convert_to_tensor(pseudo_bias), tf_convert_to_tensor(pseudolikelihood)