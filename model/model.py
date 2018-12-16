import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib import crf
from utils import rnncell as rnn
from utils.util import result_to_json, iobes_iob


class Model(object):
    def __init__(self, words_id, segs_id, labels, lengths, config, is_train=True):
        self.config = config
        self.is_train = is_train

        self.lr = config.lr
        self.char_dim = config.char_dim
        self.lstm_dim = config.lstm_dim
        self.seg_dim = config.seg_dim

        self.num_tags = config.num_tags
        self.num_chars = config.num_chars
        self.num_segs = 4

        self.initializer = initializers.xavier_initializer()
        self.global_step = tf.Variable(0, trainable=False)
        self.char_inputs = words_id
        self.seg_inputs = segs_id
        self.targets = labels

        self.dropout = config.dropout

        self.lengths = tf.cast(lengths, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = config.max_seq_length

        self.model_type = config.model_type
        self.layers = [
            {
                'dilation': 1
            },
            {
                'dilation': 1
            },
            {
                'dilation': 2
            }
        ]
        self.filter_width = 3
        self.num_filter = self.lstm_dim
        self.embedding_dim = self.char_dim + self.seg_dim
        self.repeat_times = 4
        self.cnn_output_width = 0

        embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, config)

        if self.model_type == "bilstm":
            if self.is_train:
                model_inputs = tf.nn.dropout(embedding, self.dropout)
            model_outputs = self.biLSTM_layer(model_inputs, self.lstm_dim, self.lengths)
            self.logits = self.project_layer_bilstm(model_outputs)
        elif self.model_type == "idcnn":
            if self.is_train:
                model_inputs = tf.nn.dropout(embedding, self.dropout)

            model_outputs = self.IDCNN_layer(model_inputs)
            self.logits = self.project_layer_idcnn(model_outputs)
        else:
            raise KeyError

    def get_loss(self, logits, targets, lengths):
        # print("============================================")
        # print(logits)
        # print(targets)
        # print(lengths)
        self.loss = self.loss_layer(self.logits, self.targets, self.lengths)
        with tf.variable_scope("optimizer"):
            optimizer = self.config.optimizer
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            # apply grad clip to avoid gradient explosion
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config.clip, self.config.clip), v]
                                 for g, v in grads_vars]

            self.train_op = self.opt.apply_gradients(capped_grads_vars, tf.train.get_global_step())
            # self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)
            # optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            # self.train_op = optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())

        # saver of the model
        # self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def embedding_layer(self, char_inputs, seg_inputs, config, name=None):
        embedding = []
        with tf.variable_scope("char_embedding" if not name else name, reuse=tf.AUTO_REUSE), tf.device('/cpu:0'):
            self.char_lookup = tf.get_variable(name="char_embedding",
                                               shape=[self.num_chars, self.char_dim],
                                               initializer=self.initializer)
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            if config.seg_dim:
                with tf.variable_scope("seg_embedding", reuse=tf.AUTO_REUSE), tf.device('/cpu:0'):
                    self.seg_lookup = tf.get_variable(name="seg_embedding",
                                                      shape=[self.num_tags, self.seg_dim],
                                                      initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
            embed = tf.concat(embedding, axis=-1)
        return embed


    def biLSTM_layer(self, model_inputs, lstm_dim, lengths, name=None):
        with tf.variable_scope("char_BiLSTM" if not name else name, reuse=tf.AUTO_REUSE):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholdes=True,
                        initializer=self.initializer,
                        state_is_tuple=True
                    )
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                model_inputs,
                dtype=tf.float32,
                sequence_length=lengths
            )
        return tf.concat(outputs, axis=2)


    def IDCNN_layer(self, model_inputs, name=None):
        model_inputs = tf.expand_dims(model_inputs, 2)
        reuse = False
        if not self.is_train:
            reuse = True
        with tf.variable_scope("idcnn" if not name else name, reuse=tf.AUTO_REUSE):
            filter_weights = tf.get_variable("idcnn_filter",
                                               shape=[1, self.filter_width,  self.embedding_dim,
                                                      self.num_filter],
                                               initializer=self.initializer)
            layerInput = tf.nn.conv2d(model_inputs,
                                      filter_weights,
                                      strides=[1, 1, 1, 1],
                                      padding="SAME",
                                      data_format="NHWC",
                                      name="init_layer")
            finalOutFromLayers = []
            totalWidthForLastDim = 0
            for j in range(self.repeat_times):
                for i in range(len(self.layers)):
                    dilation = self.layers[i]['dilation']
                    isLast = True if i == (len(self.layers) - 1) else False
                    with tf.variable_scope("atrous-conv-layer-%d" % i,
                                           reuse=tf.AUTO_REUSE):
                        w = tf.get_variable("filterW",
                                            shape=[1, self.filter_width, self.num_filter,
                                                   self.num_filter],
                                            initializer=tf.contrib.layers.xavier_initializer())
                        b = tf.get_variable("filterB", shape=[self.num_filter])
                        conv = tf.nn.atrous_conv2d(layerInput,
                                                   w,
                                                   rate=dilation,
                                                   padding="SAME")
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.num_filter
                        layerInput = conv

            finalOut = tf.concat(axis=3, values=finalOutFromLayers)
            keepProb = 1.0 if reuse else 0.5
            finalOut = tf.nn.dropout(finalOut, keepProb)

            finalOut = tf.squeeze(finalOut, [2])
            finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])
            self.cnn_output_width = totalWidthForLastDim
            return finalOut


    def project_layer_bilstm(self, lstm_outputs, name=None):
        with tf.variable_scope("project" if not name else name, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim * 2,
                                                self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)
                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=self.initializer)
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dype=tf.float32,
                                    initializer=self.initializer)
                b = tf.get_variable("b", shape=[self.num_tags],
                                    dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                pred = tf.nn.xw_plus_b(hidden, W, b)
            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def project_layer_idcnn(self, idcnn_outputs, name=None):
        with tf.variable_scope("project" if not name else name, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.cnn_output_width, self.num_tags],
                                    dtype=tf.float32,
                                    initializer=self.initializer)
                b = tf.get_variable("b", initializer=tf.constant(0.001,
                                                                 shape=[self.num_tags]))
                pred = tf.nn.xw_plus_b(idcnn_outputs, W, b)
        return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def loss_layer(self, project_logits, targets, lengths, name=None):
        with tf.variable_scope("crf_loss" if not name else name, reuse=tf.AUTO_REUSE):
            # small = -1000.0
            # # pad logits for crf loss
            # start_logits = tf.concat(
            #     [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
            # pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            # logits = tf.concat([project_logits, pad_logits], axis=-1)
            # logits = tf.concat([start_logits, logits], axis=1)#(?,?,8)
            # targets = tf.concat(
            #     [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), targets], axis=-1)

            self.trans = tf.get_variable("transitions",
                                         shape=[self.num_tags, self.num_tags],
                                         initializer=self.initializer)
            """Computes the log-likelihood of tag sequences in a CRF.

            Args:
              inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
                  to use as input to the CRF layer.
              tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which we
                  compute the log-likelihood.
              sequence_lengths: A [batch_size] vector of true sequence lengths.
              transition_params: A [num_tags, num_tags] transition matrix, if available.
            Returns:
              log_likelihood: A [batch_size] `Tensor` containing the log-likelihood of
                each example, given the sequence of tag indices.
              transition_params: A [num_tags, num_tags] transition matrix. This is either
                  provided by the caller or created in this function.
            """
            # Get shape information.
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=project_logits,
                tag_indices=self.targets,
                transition_params=self.trans,
                sequence_lengths=lengths
            )
            return tf.reduce_mean(-log_likelihood)


    def decode(self, logits, lengths, matrix):
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=matrix, sequence_length=lengths)
        return pred_ids