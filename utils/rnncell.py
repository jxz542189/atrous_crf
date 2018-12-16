from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging


def _get_concat_variable(name, shape, dtype, num_shards):
    sharded_variable = _get_sharded_variable(name, shape, dtype, num_shards)
    if len(sharded_variable) == 1:
        return sharded_variable[0]

    concat_name = name + "/concat"
    concat_full_name = vs.get_variable_scope().name + "/" + concat_name + ":0"
    for value in ops.get_collection(ops.GraphKeys.CONCATENATED_VARIABLES):
        if value.name == concat_full_name:
            return value
    concat_variable = array_ops.concat(sharded_variable, 0, name=concat_name)
    ops.add_to_collection(ops.GraphKeys.CONCATENATED_VARIABLES,
                          concat_variable)
    return concat_variable


def _get_sharded_variable(name, shape, dtype, num_shards):
    if num_shards > shape[0]:
        raise ValueError("Too many shards: shape=%s, num_shards=%d" %
                     (shape, num_shards))
    unit_shard_size = int(math.floor(shape[0] / num_shards))
    remaining_rows = shape[0] - unit_shard_size * num_shards

    shards = []
    for i in range(num_shards):
        current_size = unit_shard_size
        if i < remaining_rows:
            current_size += 1
        shards.append(vs.get_variable(name + "_%d" % i,
                                      [current_size] + shape[1:],
                                      dtype=dtype))
    return shards


class CoupledInputForgetGateLSTMCell(rnn_cell_impl.RNNCell):
    def __init__(self, num_units, use_peepholdes=False,
                 initializer=None, num_proj=None, proj_clip=None,
                 num_unit_shards=1, num_proj_shards=1,
                 forget_bias=1.0, state_is_tuple=True,
                 activation=math_ops.tanh, reuse=None):
        super(CoupledInputForgetGateLSTMCell, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated. Use state_is_tuple=True.", self)
        self._num_units = num_units
        self._use_peepholes = use_peepholdes
        self._initializer = initializer
        self._num_proj = num_proj
        self._proj_clip = proj_clip
        self._num_unit_shards = num_unit_shards
        self._num_proj_shards = num_proj_shards
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self._reuse = reuse

        if num_proj:
            self._state_size = (rnn_cell_impl.LSTMStateTuple(num_units, num_proj)
                                if state_is_tuple else num_units + num_proj)
            self._output_size = num_proj
        else:
            self._state_size = (rnn_cell_impl.LSTMStateTuple(num_units, num_units)
                                if state_is_tuple else 2 * num_units)
            self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, state):
        sigmoid = math_ops.sigmoid

        num_proj = self._num_units if self._num_proj is None else self._num_proj

        if self._state_is_tuple:
            c_prev, m_prev = state
        else:
            c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
            m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(2)[1]

        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        self.w_xi = tf.get_variable("_w_xi", [input_size.value, self._num_units])
        self.w_hi = tf.get_variable("_w_hi", [self._num_units, self._num_units])
        self.w_ci = tf.get_variable("_w_ci", [self._num_units, self._num_units])
        # Output gate weights
        self.w_xo = tf.get_variable("_w_xo", [input_size.value, self._num_units])
        self.w_ho = tf.get_variable("_w_ho", [self._num_units, self._num_units])
        self.w_co = tf.get_variable("_w_co", [self._num_units, self._num_units])

        # Cell weights
        self.w_xc = tf.get_variable("_w_xc", [input_size.value, self._num_units])
        self.w_hc = tf.get_variable("_w_hc", [self._num_units, self._num_units])

        # Initialize the bias vectors
        self.b_i = tf.get_variable("_b_i", [self._num_units], initializer=init_ops.zeros_initializer())
        self.b_c = tf.get_variable("_b_c", [self._num_units], initializer=init_ops.zeros_initializer())
        self.b_o = tf.get_variable("_b_o", [self._num_units], initializer=init_ops.zeros_initializer())

        i_t = sigmoid(math_ops.matmul(inputs, self.w_xi) +
                      math_ops.matmul(m_prev, self.w_hi) +
                      math_ops.matmul(c_prev, self.w_ci) +
                      self.b_i)
        c_t = ((1 - i_t) * c_prev + i_t * self._activation(math_ops.matmul(inputs, self.w_xc) +
                                                           math_ops.matmul(m_prev, self.w_hc) + self.b_c))

        o_t = sigmoid(math_ops.matmul(inputs, self.w_xo) +
                      math_ops.matmul(m_prev, self.w_ho) +
                      math_ops.matmul(c_t, self.w_co) +
                      self.b_o)

        h_t = o_t * self._activation(c_t)

        new_state = (rnn_cell_impl.LSTMStateTuple(c_t, h_t) if self._state_is_tuple else
                     array_ops.concat([c_t, h_t], 1))
        return h_t, new_state



