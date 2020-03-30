from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow.compat.v1 as tf
import os

import cProfile
from contacts_dataset import *
import parser
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
      "sequence"
    ]

needed_features = [
    'aatype',
    'deletion_probability',
    'gap_matrix',
    'hmm_profile',
#     'mutual_information',
    'non_gapped_profile',
    'num_alignments',
    'profile_with_prior',
    'profile_with_prior_without_gaps',
    'pseudo_bias',
    'pseudo_frob',
    'pseudolikelihood',
    'residue_index',
    'reweighted_profile',
    'seq_length',
    'sequence'
] #TODO

FEATURES = {
    'aatype': (tf.float32, [NUM_RES, 21]),
    'alpha_mask': (tf.int64, [NUM_RES, 1]),
    'alpha_positions': (tf.float32, [NUM_RES, 3]),
    'beta_mask': (tf.int64, [NUM_RES, 1]),
    'beta_positions': (tf.float32, [NUM_RES, 3]),
    'between_segment_residues': (tf.int64, [NUM_RES, 1]),
    'chain_name': (tf.string, [1]),
    'deletion_probability': (tf.float32, [NUM_RES, 1]),
    'domain_name': (tf.string, [1]),
    'gap_matrix': (tf.float32, [NUM_RES, NUM_RES, 1]),
    'hhblits_profile': (tf.float32, [NUM_RES, 22]),
    'hmm_profile': (tf.float32, [NUM_RES, 30]),
#     'key': (tf.string, [1]),
    'mutual_information': (tf.float32, [NUM_RES, NUM_RES, 1]),
    'non_gapped_profile': (tf.float32, [NUM_RES, 21]),
    'num_alignments': (tf.int64, [NUM_RES, 1]),
    'num_effective_alignments': (tf.float32, [1]),
    'phi_angles': (tf.float32, [NUM_RES, 1]),
    'phi_mask': (tf.int64, [NUM_RES, 1]),
    'profile': (tf.float32, [NUM_RES, 21]),
    'profile_with_prior': (tf.float32, [NUM_RES, 22]),
    'profile_with_prior_without_gaps': (tf.float32, [NUM_RES, 21]),
    'pseudo_bias': (tf.float32, [NUM_RES, 22]),
    'pseudo_frob': (tf.float32, [NUM_RES, NUM_RES, 1]),
    'pseudolikelihood': (tf.float32, [NUM_RES, NUM_RES, 484]),
    'psi_angles': (tf.float32, [NUM_RES, 1]),
    'psi_mask': (tf.int64, [NUM_RES, 1]),
    'residue_index': (tf.int64, [NUM_RES, 1]),
    'resolution': (tf.float32, [1]),
    'reweighted_profile': (tf.float32, [NUM_RES, 22]),
    'sec_structure': (tf.int64, [NUM_RES, 8]),
    'sec_structure_mask': (tf.int64, [NUM_RES, 1]),
    'seq_length': (tf.int64, [NUM_RES, 1]),
    'sequence': (tf.string, [1]),
    'solv_surf': (tf.float32, [NUM_RES, 1]),
    'solv_surf_mask': (tf.int64, [NUM_RES, 1]),
    'superfamily': (tf.string, [1]),
}

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def create_example(data, features=features):
    feature_map = {}
    for f in features:
#         print(data[f])
        if FEATURES.get(f)[0] == tf.float32:
#             print('float32')
            feature_map[f] = _float_feature(data[f].numpy())
        elif FEATURES.get(f)[0] == tf.int64:
#             print('int64')
            feature_map[f] = _int64_feature(data[f].numpy())
        elif FEATURES.get(f)[0] == tf.string:
#             print('string')
            feature_map[f] = _bytes_feature(data[f])
    example = tf.train.Example(features=tf.train.Features(feature=feature_map))
    return example

