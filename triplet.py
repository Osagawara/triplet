import tensorflow as tf
import numpy as np

from functools import reduce

class Triplet:

    def __init__(self, npy_path=None, trainable=True, dropout=0.5, alpha=0.2):
        if npy_path is not None:
            self.data_dict = np.load(npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout
        self.alpha = alpha

    def build(self, triplet, train_mode=None):
        '''
        construct the triplet network
        :param triplet: a mini-batch of triples, its shape is [batchsize, 512, 512, 3]
               the first 1/3  of the batchsize are the anchor samples
               the second 1/3 are the positive samples
               the last 1/3 are negative samples
        :param train_mode: a place holder of a bool tensor, meaning trainable or not
               now it is temporally not used
        :return: 
        '''

        # the first convolution layer, input shape is [batch, 512, 512, 3]
        # after the conv1_1, the shape is [batch, 512, 512, 64]
        # after the conv1_2, the shape is [batch, 256, 256, 64]
        # after the pooling layer, the shape is [batch, 128, 128, 64]
        self.conv1_1 = self.conv_layer(triplet, 3, [1, 2, 2, 1], 3, 64, 'conv1_1')
        # self.conv1_2 = self.conv_layer(self.conv1_1, 3, [1, 2, 2, 1], 64, 64, 'conv1_2')
        self.pool1 = self.max_pool(self.conv1_1, [1, 2, 2, 1], [1, 2, 2 ,1], 'pool1' )

        self.conv2_1 = self.conv_layer(bottom=self.pool1,
                                       filter_size=3, stride=[1, 2, 2, 1],
                                       in_channels=64, out_channels=128, name='conv2_1')
        # self.conv2_2 = self.conv_layer(self.conv2_1, 3, [1, 2, 2, 1], 128, 128, 'conv2_2')
        self.pool2 = self.max_pool(self.conv2_1, [1, 2, 2, 1], [1, 2, 2, 1], 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 3, [1, 2, 2, 1], 128, 256, 'conv3_1')
        # self.conv3_2 = self.conv_layer(self.conv3_1, 3, [1, 2, 2, 1], 256, 256, 'conv3_2')
        self.pool3 = self.max_pool(self.conv3_1, [1, 2, 2, 1], [1, 2, 2, 1], 'pool3')

        # change the input to an 4096 vector
        shape = self.pool3.shape.as_list()
        in_size = shape[1]*shape[2]*shape[3]
        self.fc4 = self.fc_layer(self.pool3, in_size=in_size, out_size=4096, name='fc4')
        self.relu4 = tf.nn.relu(self.fc4)

        # change the 4096 vector to 128 vector
        self.fc5 = self.fc_layer(self.relu4, in_size=4096, out_size=256, name='fc5')

        self.l2_norm = tf.nn.l2_normalize(self.fc5, dim=1, name='l2_norm')

        batch_size = int( triplet.shape.as_list()[0] / 3 )

        a_norm = self.l2_norm[0:batch_size]
        p_norm = self.l2_norm[batch_size:(2*batch_size)]
        n_norm = self.l2_norm[2*batch_size:(3*batch_size)]

        self.semi_loss = self.l2_loss(a_norm - p_norm) - self.l2_loss(a_norm - n_norm)
        self.hard_loss = tf.reduce_mean(tf.nn.relu(self.semi_loss + self.alpha), name='hard_loss')

    def l2_loss(self, bottom, axis=1, name='l2_loss'):
        '''
        compute the l2_loss of every sample
        :param bottom: 
        :param axis: the rank of the tensor is reduced by 1 for each entry in axis
        :param name: 
        :return: 
        '''
        return tf.reduce_sum(tf.square(bottom), axis=axis, name=name)

    def avg_pool(self, bottom, kernal, stride, name):
        '''
        create an average pooling layer
        :param bottom: the tensor to be pooled
        :param kernal: the size of the pooling kernal, it is like [1, h, w, 1]
        :param stride: the stride of the pooling kernal, it is like [1, h, w, 1] 
        :param name: 
        :return: the tensor after pooling
        '''
        return tf.nn.avg_pool(bottom, ksize=kernal, strides=stride, padding='SAME', name=name)

    def max_pool(self, bottom, kernal, stride, name):
        return tf.nn.max_pool(bottom, ksize=kernal, strides=stride, padding='SAME', name=name)

    def conv_layer(self, bottom, filter_size, stride, in_channels, out_channels, name):
        '''
        create a convolution layer
        :param bottom: the tensor to be convoluted
        :param filter_size: the size of the filter, which is an square by defalt
        :param stride: the stride of the pooling kernal, it is like [1, h, w, 1]
        :param in_channels: the channel number of bottom, the input tensor
        :param out_channels: the channel number of the output tensor
        :param name: 
        :return: 
        '''
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(filter_size, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, stride, padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./triplet-save.npy"):
        '''
        dump the variable of the net work to the file at npy_path
        :param sess: 
        :param npy_path: the file where the variables are saved
        :return: 
        '''
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count