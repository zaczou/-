import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
import math
INF = 1e30


class cudnn_gru:

    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope=None):
        self.num_layers = num_layers
        self.grus = []
        self.params = []
        self.inits = []
        self.dropout_mask = []
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(
                num_layers=1, num_units=num_units, input_size=input_size_)
            gru_bw = tf.contrib.cudnn_rnn.CudnnGRU(
                num_layers=1, num_units=num_units, input_size=input_size_)
            param_fw = tf.Variable(tf.random_uniform(
                [gru_fw.params_size()], -0.1, 0.1), validate_shape=False)
            param_bw = tf.Variable(tf.random_uniform(
                [gru_bw.params_size()], -0.1, 0.1), validate_shape=False)
            init_fw = tf.Variable(tf.zeros([1, batch_size, num_units]))
            init_bw = tf.Variable(tf.zeros([1, batch_size, num_units]))
            mask_fw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            self.grus.append((gru_fw, gru_bw, ))
            self.params.append((param_fw, param_bw, ))
            self.inits.append((init_fw, init_bw, ))
            self.dropout_mask.append((mask_fw, mask_bw, ))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        for layer in range(self.num_layers):
            gru_fw, gru_bw = self.grus[layer]
            param_fw, param_bw = self.params[layer]
            init_fw, init_bw = self.inits[layer]
            mask_fw, mask_bw = self.dropout_mask[layer]
            with tf.variable_scope("fw"):
                out_fw, _ = gru_fw(outputs[-1] * mask_fw, init_fw, param_fw)
            with tf.variable_scope("bw"):
                inputs_bw = tf.reverse_sequence(
                    outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                out_bw, _ = gru_bw(inputs_bw, init_bw, param_bw)
                out_bw = tf.reverse_sequence(
                    out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
            outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])
        return res


class native_gru:

    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope="native_gru"):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        self.scope = scope
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = tf.contrib.rnn.GRUCell(num_units)
            gru_bw = tf.contrib.rnn.GRUCell(num_units)
            init_fw = tf.Variable(tf.zeros([batch_size, num_units]))
            init_bw = tf.Variable(tf.zeros([batch_size, num_units]))
            mask_fw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            self.grus.append((gru_fw, gru_bw, ))
            self.inits.append((init_fw, init_bw, ))
            self.dropout_mask.append((mask_fw, mask_bw, ))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        outputs = [inputs]
        with tf.variable_scope(self.scope):
            for layer in range(self.num_layers):
                gru_fw, gru_bw = self.grus[layer]
                init_fw, init_bw = self.inits[layer]
                mask_fw, mask_bw = self.dropout_mask[layer]
                with tf.variable_scope("fw_{}".format(layer)):
                    out_fw, _ = tf.nn.dynamic_rnn(
                        gru_fw, outputs[-1] * mask_fw, seq_len, initial_state=init_fw, dtype=tf.float32)
                with tf.variable_scope("bw_{}".format(layer)):
                    inputs_bw = tf.reverse_sequence(
                        outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                    out_bw, _ = tf.nn.dynamic_rnn(
                        gru_bw, inputs_bw, seq_len, initial_state=init_bw, dtype=tf.float32)
                    out_bw = tf.reverse_sequence(
                        out_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        return res	


		
class SRUCell(RNNCell):
    """Simple Recurrent Unit (SRU).
       This implementation is based on:
       Tao Lei and Yu Zhang,
       "Training RNNs as Fast as CNNs,"
       https://arxiv.org/abs/1709.02755
    """

    def __init__(self, num_units, activation=None, is_train = True, reuse=None):
        self._num_units = num_units
        self._activation = activation or tf.tanh
        self._is_training = is_training
    @property
    def output_size(self):
        return self._num_units
    @property
    def state_size(self):
        return self._num_units
    def __call__(self, inputs, state, scope=None):
        """Run one step of SRU."""
        with tf.variable_scope(scope or type(self).__name__):  # "SRUCell"
            with tf.variable_scope("x_hat"):
                x = linear([inputs], self._num_units, False)
            with tf.variable_scope("gates"):
                concat = tf.sigmoid(linear([inputs], 2 * self._num_units, True))
                f, r = tf.split(concat, 2, axis = 1)
            with tf.variable_scope("candidates"):
                c = self._activation(f * state + (1 - f) * x)
                # variational dropout as suggested in the paper (disabled)
                # if self._is_training and Params.dropout is not None:
                #     c = tf.nn.dropout(c, keep_prob = 1 - Params.dropout)
            # highway connection
            # Our implementation is slightly different to the paper
            # https://arxiv.org/abs/1709.02755 in a way that highway network
            # uses x_hat instead of the cell inputs. Check equation (7) from the original
            # paper for SRU.
            h = r * c + (1 - r) * x
        return h, c		
		
		
def linear(args,
            output_size,
            bias,
            bias_initializer=None,
            kernel_initializer=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
        dtype=dtype,
        initializer=kernel_initializer)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      if bias_initializer is None:
        bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
      biases = vs.get_variable(
          _BIAS_VARIABLE_NAME, [output_size],
          dtype=dtype,
          initializer=bias_initializer)
    return nn_ops.bias_add(res, biases)		
		
		
		

def bidirectional_GRU(inputs, inputs_len, cell = None, cell_fn = tf.contrib.rnn.GRUCell, units, layers = 1, keep_prob=1.0, scope = "Bidirectional_GRU", output = 0, is_train = True, reuse = None):
    '''
    Bidirectional recurrent neural network with GRU cells.

    Args:
        inputs:     rnn input of shape (batch_size, timestep, dim)
        inputs_len: rnn input_len of shape (batch_size, )
        cell:       rnn cell of type RNN_Cell.
        output:     if 0, output returns rnn output for every timestep,
                    if 1, output returns concatenated state of backward and
                    forward rnn.
    '''
    with tf.variable_scope(scope, reuse = reuse):
        if cell is not None:
            (cell_fw, cell_bw) = cell
        else:
            shapes = inputs.get_shape().as_list()
            if len(shapes) > 3:
                inputs = tf.reshape(inputs,(shapes[0]*shapes[1],shapes[2],-1))
                inputs_len = tf.reshape(inputs_len,(shapes[0]*shapes[1],))

            # if no cells are provided, use standard GRU cell implementation
            if layers > 1:
                cell_fw = MultiRNNCell([apply_dropout(cell_fn(units), size = inputs.shape[-1] if i == 0 else units, keep_prob, is_train = is_train) for i in range(layers)])
                cell_bw = MultiRNNCell([apply_dropout(args, cell_fn(units), size = inputs.shape[-1] if i == 0 else units, keep_prob, is_train = is_train) for i in range(layers)])
            else:
                cell_fw, cell_bw = [apply_dropout(args, cell_fn(units), size = inputs.shape[-1], keep_prob, is_train = is_train) for _ in range(2)]
				
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,
                                                        sequence_length = inputs_len,
                                                        dtype=tf.float32)
        if output == 0:
            return tf.concat(outputs, 2)
        elif output == 1:
            return tf.reshape(tf.concat(states,1),(-1, shapes[1], 2*units))

			
def apply_dropout(inputs, size = None, keep_prob, is_train = True):
    if is_train:
        return tf.contrib.rnn.DropoutWrapper(inputs,
                                            output_keep_prob = 1 - keep_prob,
                                            # variational_recurrent = True,
                                            # input_size = size,
                                            dtype = tf.float32)
    else:
        return inputs

		
class ptr_net:
    def __init__(self, batch, hidden, keep_prob=1.0, is_train=None, scope="ptr_net"):
        self.gru = GRUCell(hidden)
        self.batch = batch
        self.scope = scope
        self.keep_prob = keep_prob
        self.is_train = is_train
        self.dropout_mask = dropout(tf.ones(
            [batch, hidden], dtype=tf.float32), keep_prob=keep_prob, is_train=is_train)

    def __call__(self, init, match, d, mask):
        with tf.variable_scope(self.scope):
            d_match = dropout(match, keep_prob=self.keep_prob,
                              is_train=self.is_train)
            inp, logits1 = pointer(d_match, init * self.dropout_mask, d, mask)
            d_inp = dropout(inp, keep_prob=self.keep_prob,
                            is_train=self.is_train)
            _, state = self.gru(d_inp, init)
            tf.get_variable_scope().reuse_variables()
            _, logits2 = pointer(d_match, state * self.dropout_mask, d, mask)
            return logits1, logits2


def weight(name, shape, init='he', range=None):
    """ Initializes weight.
    :param name: Variable name
    :param shape: Tensor shape
    :param init: Init mode. xavier / normal / uniform / he (default is 'he')
    :param range:
    :return: Variable
    """
    initializer = tf.constant_initializer()
    if init == 'xavier':
        fan_in, fan_out = _get_dims(shape)
        range = math.sqrt(6.0 / (fan_in + fan_out))
        initializer = tf.random_uniform_initializer(-range, range)

    elif init == 'he':
        fan_in, _ = _get_dims(shape)
        std = math.sqrt(2.0 / fan_in)
        initializer = tf.random_normal_initializer(stddev=std)

    elif init == 'normal':
        initializer = tf.random_normal_initializer(stddev=0.1)

    elif init == 'uniform':
        if range is None:
            raise ValueError("range must not be None if uniform init is used.")
        initializer = tf.random_uniform_initializer(-range, range)

    var = tf.get_variable(name, shape, initializer=initializer)
    tf.add_to_collection('l2', tf.nn.l2_loss(var))  # Add L2 Loss
    return var			
	
	
def dropout(args, keep_prob, is_train, mode="recurrent"):
    if keep_prob < 1.0:
        noise_shape = None
        scale = 1.0
        shape = tf.shape(args)
        if mode == "embedding":
            noise_shape = [shape[0], 1]
            scale = keep_prob
        if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
            noise_shape = [shape[0], 1, shape[-1]]
        args = tf.cond(is_train, lambda: tf.nn.dropout(
            args, keep_prob, noise_shape=noise_shape) * scale, lambda: args)
    return args


def softmax_mask(val, mask):
    return -INF * (1 - tf.cast(mask, tf.float32)) + val


def pointer(inputs, state, hidden, mask, scope="pointer"):
    with tf.variable_scope(scope):
        u = tf.concat([tf.tile(tf.expand_dims(state, axis=1), [
            1, tf.shape(inputs)[1], 1]), inputs], axis=2)
        s0 = tf.nn.tanh(dense(u, hidden, use_bias=False, scope="s0"))
        s = dense(s0, 1, use_bias=False, scope="s")
        s1 = softmax_mask(tf.squeeze(s, [2]), mask)
        a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        res = tf.reduce_sum(a * inputs, axis=1)
        return res, s1


def summ(memory, hidden, mask, keep_prob=1.0, is_train=None, scope="summ"):
    with tf.variable_scope(scope):
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        s0 = tf.nn.tanh(dense(d_memory, hidden, scope="s0"))
        s = dense(s0, 1, use_bias=False, scope="s")
        s1 = softmax_mask(tf.squeeze(s, [2]), mask)
        a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        res = tf.reduce_sum(a * memory, axis=1)
        return res


def dot_attention(inputs, memory, mask, hidden, keep_prob=1.0, is_train=None, scope="dot_attention"):
    with tf.variable_scope(scope):

        d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        JX = tf.shape(inputs)[1]

        with tf.variable_scope("attention"):
            inputs_ = tf.nn.relu(
                dense(d_inputs, hidden, use_bias=False, scope="inputs"))
            memory_ = tf.nn.relu(
                dense(d_memory, hidden, use_bias=False, scope="memory"))
            outputs = tf.matmul(inputs_, tf.transpose(
                memory_, [0, 2, 1])) / (hidden ** 0.5)
            mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
            logits = tf.nn.softmax(softmax_mask(outputs, mask))
            outputs = tf.matmul(logits, memory)
            res = tf.concat([inputs, outputs], axis=2)

        with tf.variable_scope("gate"):
            dim = res.get_shape().as_list()[-1]
            d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
            gate = tf.nn.sigmoid(dense(d_res, dim, use_bias=False))
            return res * gate


def dense(inputs, hidden, use_bias=True, scope="dense"):
    with tf.variable_scope(scope):
        shape = tf.shape(inputs)
        dim = inputs.get_shape().as_list()[-1]
        out_shape = [shape[idx] for idx in range(
            len(inputs.get_shape().as_list()) - 1)] + [hidden]
        flat_inputs = tf.reshape(inputs, [-1, dim])
        W = tf.get_variable("W", [dim, hidden])
        res = tf.matmul(flat_inputs, W)
        if use_bias:
            b = tf.get_variable(
                "b", [hidden], initializer=tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, b)
        res = tf.reshape(res, out_shape)
        return res
