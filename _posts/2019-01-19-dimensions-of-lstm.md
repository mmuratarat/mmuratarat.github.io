---
layout: post
title: "Dimensions of matrices in an LSTM Block"
author: "MMA"
comments: true
---

A general LSTM block can be shown as given below.

![Placeholder Image](https://raw.githubusercontent.com/mmuratarat/mmuratarat.github.io/master/_posts/images/lstm.png)

If we write down the equations, we will have:

1. Input gate:
$$ i_{t} = \sigma (W_{xi}X_{t} + W_{hi}h_{t-1} + b_{i})$$

2. Forget gate:
$$ f_{t} = \sigma (W_{xf}X_{t} + W_{hf}h_{t-1} + b_{f})$$

3. New Candidate:
$$ \widetilde{C}_{t} = tanh (W_{xc}X_{t} + W_{hc}h_{t-1} + b_{c})$$

4. Cell State:
$$ C_{t} = f_{t}\circ C_{t-1} + i_{t}  \circ \widetilde{C}_{t}$$

5. Output gate:
$$ o_{t} = \sigma (W_{xo}X_{t} + W_{ho}h_{t-1} + b_{o})$$

6. Hidden State:
$$ h_{t} = o_{t}\circ tanh(C_{t})$$

Here, $\circ$ represents the Hadamard product (elementwise product).

We can concatenate the weight matrices for $X_{t}$ and $h_{t-1}$ horizontally, we can rewrite the equations above as the following:

1. Input gate:
$$ i_{t} = \sigma (W_{i} \cdot [X_{t}, h_{t-1}] + b_{i})$$

2. Forget gate:
$$ f_{t} = \sigma (W_{f} \cdot [X_{t}, h_{t-1}] + b_{f})$$

3. New Candidate:
$$ \widetilde{C}_{t} = tanh (W_{c} \cdot [X_{t}, h_{t-1}] + b_{c})$$

4. Cell State:
$$ C_{t} = f_{t}\circ C_{t-1} + i_{t}  \circ \widetilde{C}_{t}$$

5. Output gate:
$$ o_{t} = \sigma (W_{o} \cdot [X_{t}, h_{t-1}] + b_{o})$$

6. Hidden State:
$$ h_{t} = o_{t}\circ tanh(C_{t})$$

# Tensorflow Dimensions
In Tensorflow, LSTM variables are defined in `LSTMCell.build` method. The source code can be found in [rnn_cell_impl.py](https://github.com/tensorflow/tensorflow/blob/f52351444551016d7dd949a5aa599da489a97045/tensorflow/python/ops/rnn_cell_impl.py#L728){:target="_blank"}:

{% highlight python %} 
self._kernel = self.add_variable(
    _WEIGHTS_VARIABLE_NAME,
    shape=[input_depth + h_depth, 4 * self._num_units],
    initializer=self._initializer,
    partitioner=maybe_partitioner)
self._bias = self.add_variable(
    _BIAS_VARIABLE_NAME,
    shape=[4 * self._num_units],
    initializer=init_ops.zeros_initializer(dtype=self.dtype))
{% endhighlight %}

As one can see easily, there's just one `[input_depth + h_depth, 4 * self._num_units]` shaped weight matrix and `[4 * self._num_units]` shaped bias vector, not 8 different matrices for weights and 4 different vectors for biases, and all of them are multiplied simultaneously in a batch.

The gates are defined this way:

{% highlight python %} 
# i = input_gate, j = new_input, f = forget_gate, o = output_gate
i, j, f, o = array_ops.split(value=gate_inputs, num_or_size_splits=4, axis=one)
{% endhighlight %}

Considering that we have a data, shape of `[batch_size, time_steps, number_features]`, $X_{t}$ is the input of time-step $t$ which is an array with the shape of `[batch_size, num_features]`, $h_{t-1}$ is the hidden state of previous time-step which is an array with the shape of `[batch_size, num_units]`, and $C_{t-1}$ is the cell state of previous time-step, which is an array with the shape of `[batch_size, num_units]`. In that case, Tensorflow will concatenate inputs ($X_{t}$) and hidden state ($h_{t-1}$) by column and multiple it with kernel (weight) matrix that we mentioned previously. For more info, look at [here](https://github.com/tensorflow/tensorflow/blob/f52351444551016d7dd949a5aa599da489a97045/tensorflow/python/ops/rnn_cell_impl.py#L763).

Each of the $W_{xi}$, $W_{xf}$, $W_{xc}$ and $W_{xo}$, is an array with the shape of `[num_features, num_units]` and, similarly, each of the $W_{hi}$, $W_{hf}$, $W_{hc}$ and $W_{ho}$ is an array with the shape of `[num_units, num_units]`. If we first concatenate each gate weight matrices, corresponding to input and hidden state, vertically, we will have separate $W_{i}$, $W_{c}$, $W_{f}$ and $W_{o}$ matrices, which each will have the shape of `[num_features + num_units, num_units]`. Then, if we concatenate $W_{i}$,  $W_{c}$, $W_{f}$ and $W_{o}$ matrices horizontally, we will have kernel (weights) matrix, given by Tensorflow, which has shape `[num_features + num_units, 4 * num_units]`.

**NOTE**: Tensorflow uses the letter `j` to denote new input (candidate), we use the letter `c`.

# Mathematical Representation
Let's denote $B$ as batch size, $F$ as number of features and $U$ as number of units in an LSTM block, therefore, the dimensions will be computed as follows:

$X_{t} \in \mathbb{R}^{B \times F}$

$h_{t-1} \in \mathbb{R}^{B \times U}$

$h_{t} \in \mathbb{R}^{B \times U}$

$C_{t-1} \in \mathbb{R}^{B \times U}$

$W_{xi} \in \mathbb{R}^{F \times U}$

$W_{xf} \in \mathbb{R}^{F \times U}$

$W_{xc} \in \mathbb{R}^{F \times U}$

$W_{xo} \in \mathbb{R}^{F \times U}$

$W_{hi} \in \mathbb{R}^{U \times U}$

$W_{hf} \in \mathbb{R}^{U \times U}$

$W_{hc} \in \mathbb{R}^{U \times U}$

$W_{ho} \in \mathbb{R}^{U \times U}$

$W_{i} \in \mathbb{R}^{F+U \times U}$

$W_{c} \in \mathbb{R}^{F+U \times U}$

$W_{f} \in \mathbb{R}^{F+U \times U}$ 

$W_{o} \in \mathbb{R}^{F+U \times U}$ 

$b_{i} \in \mathbb{R}^{U}$

$b_{c} \in \mathbb{R}^{U}$

$b_{f} \in \mathbb{R}^{U}$

$b_{o} \in \mathbb{R}^{U}$

$i_{t} \in \mathbb{R}^{B \times U}$

$f_{t} \in \mathbb{R}^{B \times U}$

$C_{t} \in \mathbb{R}^{B \times U}$

$h_{t} \in \mathbb{R}^{B \times U}$

$o_{t} \in \mathbb{R}^{B \times U}$

**NOTE**: Batch size can be $1$. In that case, $B=1$.

# LSTM with Peephole Connections
In a basic LSTM cell, the gate controllers can look only at the input $X_{t}$, and the previous short-term state $h_{t-1}$. It may be a good idea to give them a bit more context by letting them peek at the long-term state as well. This LSTM variant with extra connections called peephole connections which lead from the cells to the gates: the previous long-term state $C_{t-1}$ is added as an input to the controllers of the forget gate and input gate, and the current long-term state $C_{t}$ is added to the controller of the output gate.

![Placeholder Image](https://raw.githubusercontent.com/mmuratarat/mmuratarat.github.io/master/_posts/images/lstm_peepholes.png)

Based on the information above, if we write down the equations, we will have:

1. Input gate:
$$ i_{t} = \sigma (W_{xi}X_{t} + W_{hi}h_{t-1} + W_{ci} \circ C_{t-1} + b_{i})$$

2. Forget gate:
$$ f_{t} = \sigma (W_{xf}X_{t} + W_{hf}h_{t-1} + W_{cf} \circ C_{t-1} + b_{f})$$

3. New Candidate:
$$ \widetilde{C}_{t} = tanh (W_{xc}X_{t} + W_{hc}h_{t-1} + b_{c})$$

4. Cell State:
$$ C_{t} = f_{t}\circ C_{t-1} + i_{t}  \circ \widetilde{C}_{t}$$

5. Output gate:
$$ o_{t} = \sigma (W_{xo}X_{t} + W_{ho}h_{t-1} + W_{co} \circ C_{t} + b_{o})$$

6. Hidden State:
$$ h_{t} = o_{t}\circ tanh(C_{t})$$

Here, $\circ$ represents the Hadamard product (elementwise product).

Tensorflow implementation is given as follows:
{% highlight python %} 
      # Diagonal connections
      if self._use_peepholes:
        w_f_diag = vs.get_variable(
            "W_F_diag", shape=[self._num_units], dtype=dtype)
        w_i_diag = vs.get_variable(
            "W_I_diag", shape=[self._num_units], dtype=dtype)
        w_o_diag = vs.get_variable(
            "W_O_diag", shape=[self._num_units], dtype=dtype)
      if self._use_peepholes:
        c = (sigmoid(f + 1 + w_f_diag * c_prev) * c_prev +
             sigmoid(i + w_i_diag * c_prev) * tanh(j))
      else:
        c = (sigmoid(f + 1) * c_prev + sigmoid(i) * tanh(j))
{% endhighlight %}

As one can easily see that, the weights for peephole connections are one dimensional array, shape of `[num_units]`.
