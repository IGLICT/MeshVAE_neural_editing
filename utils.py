import tensorflow as tf
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial.distance

from tensorflow.python.framework import ops
'''
spectral conv code from https://github.com/mdeff/cnn_graph
'''



def spectral_conv(x, L, Fout, K,W,name='graph_conv',activation='tanh'):
    with tf.variable_scope(name) as scope:
        N, M, Fin = x.get_shape()
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, -1])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, M, Fin, -1])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [-1, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        x = tf.matmul(x, W)  # N*M x Fout
        x = tf.reshape(x, [-1, M, Fout])
        if activation=='tanh':
            x = tf.tanh(x)
        elif activation == 'selu':
            x = tf.keras.activations.selu(x)
        elif activation =='spp':
            x = self.softplusplus(x)
        elif activation =='lrelu':
            x = self.leaky_relu(x)
        elif activation =='none':
            x = x
        return x  # N x M x Fout

def Laplacian(W, normalized=True):
    """Return the Laplacian of the weigth matrix."""

    # Degree matrix.
    d = W.sum(axis=0)
    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        #d += np.spacing(np.array(0.0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr_matrix
    return L
def rescale_L(L, lmax=2):
    """Rescale the Laplacian eigenvalues in [-1,1]."""
    M, M = L.shape
    I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
    L /= lmax / 2
    L -= I
    return L


def leaky_relu(input_, alpha = 0.02):
    return tf.maximum(input_, alpha*input_)

def softplusplus(input_, alpha=0.02):
    return tf.log(1.0+tf.exp(input_*(1.0-alpha)))+alpha*input_-tf.log(2.0)

def linear(input_, input_size, output_size, name='Linear', stddev=0.02, bias_start=0.0):
    with tf.variable_scope(name) as scope:
        matrix = tf.get_variable("weights", [input_size, output_size], tf.float32,
                tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [output_size], tf.float32, 
          initializer=tf.constant_initializer(bias_start))

        return tf.matmul(input_, matrix) + bias

def batch_norm_wrapper(inputs, name = 'batch_norm',is_training = False, decay = 0.9, epsilon = 1e-5):
    with tf.variable_scope(name) as scope:
        if is_training == True:
            scale = tf.get_variable('scale', dtype=tf.float32, trainable=True, initializer=tf.ones([inputs.get_shape()[-1]],dtype=tf.float32))  
            beta = tf.get_variable('beta', dtype=tf.float32, trainable=True, initializer=tf.zeros([inputs.get_shape()[-1]],dtype=tf.float32))
            pop_mean = tf.get_variable('overallmean',  dtype=tf.float32,trainable=False, initializer=tf.zeros([inputs.get_shape()[-1]],dtype=tf.float32))
            pop_var = tf.get_variable('overallvar',  dtype=tf.float32, trainable=False, initializer=tf.ones([inputs.get_shape()[-1]],dtype=tf.float32))
        else:
            scope.reuse_variables()
            scale = tf.get_variable('scale', dtype=tf.float32, trainable=True)
            beta = tf.get_variable('beta', dtype=tf.float32, trainable=True)
            pop_mean = tf.get_variable('overallmean', dtype=tf.float32, trainable=False)
            pop_var = tf.get_variable('overallvar', dtype=tf.float32, trainable=False)

        if is_training:
            axis = list(range(len(inputs.get_shape()) - 1))
            batch_mean, batch_var = tf.nn.moments(inputs,axis)
            train_mean = tf.assign(pop_mean,pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,batch_mean, batch_var, beta, scale, epsilon)
        else:
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)






def binarize(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()

    with ops.name_scope("Binarized") as name:
        with g.gradient_override_map({"Sign": "Identity"}):
            return tf.sign(x)

def gaussian(batch_size, n_dim, mean=0, var=1, n_labels=10, use_label_info=False):
    if use_label_info:
        if n_dim != 2:
            raise Exception("n_dim must be 2.")

        def sample(n_labels):
            x, y = np.random.normal(mean, var, (2,))
            angle = np.angle((x - mean) + 1j * (y - mean), deg=True)

            label = (int(n_labels * angle)) // 360

            if label < 0:
                label += n_labels

            return np.array([x, y]).reshape((2,)), label

        z = np.empty((batch_size, n_dim), dtype=np.float32)
        z_id = np.empty((batch_size, 1), dtype=np.int32)
        for batch in xrange(batch_size):
            for zi in xrange(int(n_dim / 2)):
                a_sample, a_label = sample(n_labels)
                z[batch, zi * 2:zi * 2 + 2] = a_sample
                z_id[batch] = a_label
        return z, z_id
    else:
        z = np.random.normal(mean, var, (batch_size, n_dim)).astype(np.float32)
        return z