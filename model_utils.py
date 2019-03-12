# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Model utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf
from model_utils import variable_mapping

np.random.seed(0)

FLAGS = tf.app.flags.FLAGS


def generate_mask_sub_sentence(hparams, input_x_id, id_to_word):
  """mask a sub sentence"""
  p = np.full([FLAGS.batch_size, FLAGS.sequence_length], True, dtype=bool)
  input_x = [' '.join([id_to_word[char_id] for char_id in input_x_id[batch]]) for batch in range(FLAGS.batch_size)]
  eos_idx = [list(input_x_id[batch]).index(hparams.eos_idx) if hparams.eos_idx in list(input_x_id[batch]) else FLAGS.sequence_length-1 for batch in range(FLAGS.batch_size)]
  for b in range(FLAGS.batch_size):
    x = input_x[b]
    if '<eos>' in x:
      x = x[:x.index('<eos>')]
    x = x.replace('，', ',')
    x = x.replace('。', ',')
    x = x.replace('!', ',')
    x = x.replace('！', ',')
    sep_x = x.split(',')
    if len(sep_x) <= 3:  # random mask
      masked_length = int((1 - FLAGS.is_present_rate) * FLAGS.sequence_length) - 1
      start_mask_idx = np.random.randint(2, int(FLAGS.sequence_length / 4))
      p[b, start_mask_idx:start_mask_idx + masked_length] = False

      # code below have bug
      # start_mask_idx = np.random.randint(2,  max(3, int(eos_idx[b]/2)))
      # masked_length = int((1-FLAGS.is_present_rate) * eos_idx[b]) - 1
      # end_mask_idx = min(start_mask_idx+masked_length, FLAGS.sequence_length)
      #
      # p[b, start_mask_idx:end_mask_idx] =  False
    else:
      max_len, max_sub_sen = max([(len(sub_sen.strip().split()), sub_sen) for sub_sen in sep_x[1:-2]])  # not include first and last sub sentence
      max_len_sub_sen_idx = sep_x.index(max_sub_sen)

      # start_mask_idx = sum([len(sub_sen.split())+1 for sub_sen in sep_x[:max_len_sub_sen_idx]]) - 1
      # end_mask_idx = start_mask_idx + len(max_sub_sen.split()) + 1

      # make sure start mask index >= 1
      start_mask_idx = max(sum([len(sub_sen.split()) + 1 for sub_sen in sep_x[:max_len_sub_sen_idx]]) - 2, 1)
      # make sure end mask index <= seq_length-1
      end_mask_idx = min(start_mask_idx + len(max_sub_sen.split()) + 1, FLAGS.sequence_length - 1)

      p[b, start_mask_idx:end_mask_idx+1] = False
      # print(input_x[b])
      # print(start_mask_idx)
      # print(end_mask_idx)
      # print(p[b])
      # exit()

  return p


def generate_mask_sub_sentence_old(input_x_id, id_to_word):
  """mask a sub sentence"""
  p = np.full([FLAGS.batch_size, FLAGS.sequence_length], True, dtype=bool)
  input_x = [' '.join([id_to_word[char_id] for char_id in input_x_id[batch]]) for batch in range(FLAGS.batch_size)]
  for b in range(FLAGS.batch_size):
    x = input_x[b]
    if '<eos>' in x:
      x = x[:x.index('<eos>')]
    x = x.replace('，', ',')
    x = x.replace('。', ',')
    x = x.replace('!', ',')
    x = x.replace('！', ',')
    sep_x = x.split(',')
    if len(sep_x) <= 3:  # random mask
      masked_length = int((1 - FLAGS.is_present_rate) * FLAGS.sequence_length) - 1
      start_mask_idx = np.random.randint(2, int(FLAGS.sequence_length / 4))
      p[b, start_mask_idx:start_mask_idx+masked_length] =  False
    else:
      max_len, max_sub_sen = max([(len(sub_sen.strip().split()), sub_sen) for sub_sen in sep_x[1:-2]])  # not include first and last sub sentence
      max_len_sub_sen_idx = sep_x.index(max_sub_sen)
      # start_mask_idx = sum([len(sub_sen.split())+1 for sub_sen in sep_x[:max_len_sub_sen_idx]]) - 1
      # end_mask_idx = start_mask_idx + len(max_sub_sen.split()) + 1

      # make sure start mask index >= 1
      start_mask_idx = max(sum([len(sub_sen.split()) + 1 for sub_sen in sep_x[:max_len_sub_sen_idx]]) - 2, 1)
      # make sure end mask index <= seq_length-1
      end_mask_idx = min(start_mask_idx + len(max_sub_sen.split()) + 1, FLAGS.sequence_length-1)

      p[b, start_mask_idx:end_mask_idx+1] = False
      # print(input_x[b])
      # print(start_mask_idx)
      # print(end_mask_idx)
      # print(p[b])
      # exit()

  return p


def generate_mask(hparams, input_x_id):
    """Generate the mask to be fed into the model."""
    if FLAGS.mask_strategy == 'random':
        mask_p = np.random.choice(
            [True, False],
            size=[FLAGS.batch_size, int(FLAGS.sequence_length * 2 / 3)],
            p=[FLAGS.is_present_rate, 1. - FLAGS.is_present_rate])

        p = np.full([FLAGS.batch_size, FLAGS.sequence_length], True, dtype=bool)

        # Create contiguous masked section to be False.
        for i, p_value in enumerate(mask_p):
            p[i, :int(FLAGS.sequence_length * 2 / 3)] = p_value

    elif FLAGS.mask_strategy == 'contiguous':
        p = np.full([FLAGS.batch_size, FLAGS.sequence_length], True, dtype=bool)
        eos_idx = [list(input_x_id[batch]).index(hparams.eos_idx) if hparams.eos_idx in list(
            input_x_id[batch]) else FLAGS.sequence_length-1 for batch in range(FLAGS.batch_size)]
        for b in range(FLAGS.batch_size):
            start_mask_idx = np.random.randint(2, max(3, int(eos_idx[b] / 2)))
            masked_length = int((1 - FLAGS.is_present_rate) * eos_idx[b]) - 1
            end_mask_idx = min(start_mask_idx + masked_length, FLAGS.sequence_length)

            p[b, start_mask_idx:end_mask_idx] = False

    else:
        raise NotImplementedError

    return p


def generate_mask_old():
  """Generate the mask to be fed into the model."""
  if FLAGS.mask_strategy == 'random':
    p = np.random.choice(
        [True, False],
        size=[FLAGS.batch_size, FLAGS.sequence_length],
        p=[FLAGS.is_present_rate, 1. - FLAGS.is_present_rate])

  elif FLAGS.mask_strategy == 'contiguous':
    masked_length = int((1 - FLAGS.is_present_rate) * FLAGS.sequence_length) - 1
    # Determine location to start masking.
    start_mask = np.random.randint(
        1, FLAGS.sequence_length - masked_length + 1, size=FLAGS.batch_size)
    p = np.full([FLAGS.batch_size, FLAGS.sequence_length], True, dtype=bool)

    # Create contiguous masked section to be False.
    for i, index in enumerate(start_mask):
      p[i, index:index + masked_length] = False

  else:
    raise NotImplementedError

  return p


def assign_percent_real(session, percent_real_update, new_rate, current_rate):
  """Run assign operation where the we load the current_rate of percent
  real into a Tensorflow variable.

  Args:
    session:  Current tf.Session.
    percent_real_update: tf.assign operation.
    new_rate: tf.placeholder for the new rate.
    current_rate: Percent of tokens that are currently real.  Fake tokens
      are the ones being imputed by the Generator.
  """
  session.run(percent_real_update, feed_dict={new_rate: current_rate})


def assign_learning_rate(session, lr_update, lr_placeholder, new_lr):
  """Run assign operation where the we load the current_rate of percent
  real into a Tensorflow variable.

  Args:
    session:  Current tf.Session.
    lr_update: tf.assign operation.
    lr_placeholder: tf.placeholder for the new learning rate.
    new_lr: New learning rate to use.
  """
  session.run(lr_update, feed_dict={lr_placeholder: new_lr})


def clip_weights(variables, c_lower, c_upper):
  """Clip a list of weights to be within a certain range.

  Args:
    variables: List of tf.Variable weights.
    c_lower: Lower bound for weights.
    c_upper: Upper bound for weights.
  """
  clip_ops = []

  for var in variables:
    clipped_var = tf.clip_by_value(var, c_lower, c_upper)

    clip_ops.append(tf.assign(var, clipped_var))
  return tf.group(*clip_ops)


def retrieve_init_savers_old(hparams):
  """Retrieve a dictionary of all the initial savers for the models.

  Args:
    hparams:  MaskGAN hyperparameters.
  """
  ## Dictionary of init savers.
  init_savers = {}

  ## Load Generator weights from MaskGAN checkpoint.
  if FLAGS.maskgan_ckpt:
    gen_vars = [
        v for v in tf.trainable_variables() if v.op.name.startswith('gen')
    ]
    init_saver = tf.train.Saver(var_list=gen_vars)
    init_savers['init_saver'] = init_saver

    ## Load the Discriminator weights from the MaskGAN checkpoint if
    # the weights are compatible.
    if FLAGS.discriminator_model == 'seq2seq_vd':
      dis_variable_maps = variable_mapping.dis_seq2seq_vd(hparams)
      dis_init_saver = tf.train.Saver(var_list=dis_variable_maps)
      init_savers['dis_init_saver'] = dis_init_saver

  ## Load weights from language model checkpoint.
  if FLAGS.language_model_ckpt_dir:
    if FLAGS.maskgan_ckpt is None:
      ## Generator Variables/Savers.
      if FLAGS.generator_model == 'rnn_nas':
        gen_variable_maps = variable_mapping.rnn_nas(hparams, model='gen')
        gen_init_saver = tf.train.Saver(var_list=gen_variable_maps)
        init_savers['gen_init_saver'] = gen_init_saver

      elif FLAGS.generator_model == 'seq2seq_nas':
        # Encoder.
        gen_encoder_variable_maps = variable_mapping.gen_encoder_seq2seq_nas(
            hparams)
        gen_encoder_init_saver = tf.train.Saver(
            var_list=gen_encoder_variable_maps)
        # Decoder.
        gen_decoder_variable_maps = variable_mapping.gen_decoder_seq2seq_nas(
            hparams)
        gen_decoder_init_saver = tf.train.Saver(
            var_list=gen_decoder_variable_maps)
        init_savers['gen_encoder_init_saver'] = gen_encoder_init_saver
        init_savers['gen_decoder_init_saver'] = gen_decoder_init_saver

      # seq2seq_vd derived from the same code base as seq2seq_zaremba.
      elif (FLAGS.generator_model == 'seq2seq_zaremba' or
            FLAGS.generator_model == 'seq2seq_vd'):
        # Encoder.
        gen_encoder_variable_maps = variable_mapping.gen_encoder_seq2seq(
            hparams)
        gen_encoder_init_saver = tf.train.Saver(
            var_list=gen_encoder_variable_maps)
        # Decoder.
        gen_decoder_variable_maps = variable_mapping.gen_decoder_seq2seq(
            hparams)
        gen_decoder_init_saver = tf.train.Saver(
            var_list=gen_decoder_variable_maps)
        init_savers['gen_encoder_init_saver'] = gen_encoder_init_saver
        init_savers['gen_decoder_init_saver'] = gen_decoder_init_saver

      else:
        raise NotImplementedError

    ## Discriminator Variables/Savers.
    if FLAGS.discriminator_model == 'rnn_nas':
      dis_variable_maps = variable_mapping.rnn_nas(hparams, model='dis')
      dis_init_saver = tf.train.Saver(var_list=dis_variable_maps)
      init_savers['dis_init_saver'] = dis_init_saver

    # rnn_vd derived from the same code base as rnn_zaremba.
    elif (FLAGS.discriminator_model == 'rnn_zaremba' or
          FLAGS.discriminator_model == 'rnn_vd'):
      dis_variable_maps = variable_mapping.rnn_zaremba(hparams, model='dis')
      dis_init_saver = tf.train.Saver(var_list=dis_variable_maps)
      init_savers['dis_init_saver'] = dis_init_saver

    elif (FLAGS.discriminator_model == 'bidirectional_zaremba' or
          FLAGS.discriminator_model == 'bidirectional_vd'):
      dis_fwd_variable_maps = variable_mapping.dis_fwd_bidirectional(hparams)
      dis_bwd_variable_maps = variable_mapping.dis_bwd_bidirectional(hparams)
      # Savers for the forward/backward Discriminator components.
      dis_fwd_init_saver = tf.train.Saver(var_list=dis_fwd_variable_maps)
      dis_bwd_init_saver = tf.train.Saver(var_list=dis_bwd_variable_maps)
      init_savers['dis_fwd_init_saver'] = dis_fwd_init_saver
      init_savers['dis_bwd_init_saver'] = dis_bwd_init_saver

    elif FLAGS.discriminator_model == 'cnn':
      dis_variable_maps = variable_mapping.cnn()
      dis_init_saver = tf.train.Saver(var_list=dis_variable_maps)
      init_savers['dis_init_saver'] = dis_init_saver

    elif FLAGS.discriminator_model == 'seq2seq_vd':
      # Encoder.
      dis_encoder_variable_maps = variable_mapping.dis_encoder_seq2seq(hparams)
      dis_encoder_init_saver = tf.train.Saver(
          var_list=dis_encoder_variable_maps)
      # Decoder.
      dis_decoder_variable_maps = variable_mapping.dis_decoder_seq2seq(hparams)
      dis_decoder_init_saver = tf.train.Saver(
          var_list=dis_decoder_variable_maps)
      init_savers['dis_encoder_init_saver'] = dis_encoder_init_saver
      init_savers['dis_decoder_init_saver'] = dis_decoder_init_saver

  return init_savers


def retrieve_init_savers(hparams):
  """Retrieve a dictionary of all the initial savers for the models.

  Args:
    hparams:  MaskGAN hyperparameters.
  """
  if FLAGS.maskgan_ckpt:
    ## Dictionary of init savers.
    vars_to_restore = [v for v in tf.trainable_variables()]
    init_savers = tf.train.Saver(var_list=vars_to_restore)

    return init_savers
  else:
    assert('No maskgan checkpoint to load!!!')

def init_fn_old(init_savers, sess):
  """The init_fn to be passed to the Supervisor.

  Args:
    init_savers:  Dictionary of init_savers.  'init_saver_name': init_saver.
    sess:  tf.Session.
  """
  ## Load Generator weights from MaskGAN checkpoint.
  if FLAGS.maskgan_ckpt:
    print('Restoring Generator from %s.' % FLAGS.maskgan_ckpt)
    tf.logging.info('Restoring Generator from %s.' % FLAGS.maskgan_ckpt)
    print('Asserting Generator is a seq2seq-variant.')
    tf.logging.info('Asserting Generator is a seq2seq-variant.')
    assert FLAGS.generator_model.startswith('seq2seq')
    init_saver = init_savers['init_saver']
    init_saver.restore(sess, FLAGS.maskgan_ckpt)

    ## Load the Discriminator weights from the MaskGAN checkpoint if
    # the weights are compatible.
    if FLAGS.discriminator_model == 'seq2seq_vd':
      print('Restoring Discriminator from %s.' % FLAGS.maskgan_ckpt)
      tf.logging.info('Restoring Discriminator from %s.' % FLAGS.maskgan_ckpt)
      dis_init_saver = init_savers['dis_init_saver']
      dis_init_saver.restore(sess, FLAGS.maskgan_ckpt)

  ## Load weights from language model checkpoint.
  if FLAGS.language_model_ckpt_dir:
    if FLAGS.maskgan_ckpt is None:
      ## Generator Models.
      if FLAGS.generator_model == 'rnn_nas':
        load_ckpt = tf.train.latest_checkpoint(FLAGS.language_model_ckpt_dir)
        print('Restoring Generator from %s.' % load_ckpt)
        tf.logging.info('Restoring Generator from %s.' % load_ckpt)
        gen_init_saver = init_savers['gen_init_saver']
        gen_init_saver.restore(sess, load_ckpt)

      elif FLAGS.generator_model.startswith('seq2seq'):
        load_ckpt = tf.train.latest_checkpoint(FLAGS.language_model_ckpt_dir)
        print('Restoring Generator from %s.' % load_ckpt)
        tf.logging.info('Restoring Generator from %s.' % load_ckpt)
        gen_encoder_init_saver = init_savers['gen_encoder_init_saver']
        gen_decoder_init_saver = init_savers['gen_decoder_init_saver']
        gen_encoder_init_saver.restore(sess, load_ckpt)
        gen_decoder_init_saver.restore(sess, load_ckpt)

    ## Discriminator Models.
    if (FLAGS.discriminator_model == 'rnn_nas' or
        FLAGS.discriminator_model == 'rnn_zaremba' or
        FLAGS.discriminator_model == 'rnn_vd' or
        FLAGS.discriminator_model == 'cnn'):
      load_ckpt = tf.train.latest_checkpoint(FLAGS.language_model_ckpt_dir)
      print('Restoring Discriminator from %s.' % load_ckpt)
      tf.logging.info('Restoring Discriminator from %s.' % load_ckpt)
      dis_init_saver = init_savers['dis_init_saver']
      dis_init_saver.restore(sess, load_ckpt)

    elif (FLAGS.discriminator_model == 'bidirectional_zaremba' or
          FLAGS.discriminator_model == 'bidirectional_vd'):
      assert FLAGS.language_model_ckpt_dir_reversed is not None, (
          'Need a reversed directory to fill in the backward components.')
      load_fwd_ckpt = tf.train.latest_checkpoint(FLAGS.language_model_ckpt_dir)
      load_bwd_ckpt = tf.train.latest_checkpoint(
          FLAGS.language_model_ckpt_dir_reversed)
      print('Restoring Discriminator from %s and %s.' % (load_fwd_ckpt,
                                                         load_bwd_ckpt))
      tf.logging.info('Restoring Discriminator from %s and %s.' %
                      (load_fwd_ckpt, load_bwd_ckpt))
      dis_fwd_init_saver = init_savers['dis_fwd_init_saver']
      dis_bwd_init_saver = init_savers['dis_bwd_init_saver']
      dis_fwd_init_saver.restore(sess, load_fwd_ckpt)
      dis_bwd_init_saver.restore(sess, load_bwd_ckpt)

    elif FLAGS.discriminator_model == 'seq2seq_vd':
      load_ckpt = tf.train.latest_checkpoint(FLAGS.language_model_ckpt_dir)
      print('Restoring Discriminator from %s.' % load_ckpt)
      tf.logging.info('Restoring Discriminator from %s.' % load_ckpt)
      dis_encoder_init_saver = init_savers['dis_encoder_init_saver']
      dis_decoder_init_saver = init_savers['dis_decoder_init_saver']
      dis_encoder_init_saver.restore(sess, load_ckpt)
      dis_decoder_init_saver.restore(sess, load_ckpt)

  else:
    return


def init_fn(init_savers, sess):
  """The init_fn to be passed to the Supervisor.

  Args:
    init_savers:  Dictionary of init_savers.  'init_saver_name': init_saver.
    sess:  tf.Session.
  """
  ## Load Generator weights from MaskGAN checkpoint.
  if FLAGS.maskgan_ckpt:
    print('Restoring Generator and Discriminator from %s.' % FLAGS.maskgan_ckpt)
    tf.logging.info('Restoring Generator and Discriminator from %s.\n' % FLAGS.maskgan_ckpt)
    print('Asserting Generator and Discriminator are a seq2seq-variant.')
    tf.logging.info('Asserting Generator and Discriminator are a seq2seq-variant.\n')
    assert FLAGS.generator_model.startswith('seq2seq')
    init_savers.restore(sess, FLAGS.maskgan_ckpt)
