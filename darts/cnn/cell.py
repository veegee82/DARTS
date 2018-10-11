import tensorflow as tf
from tfcore.utilities.utils import reduce_std
from cnn.node import Node


class Cell(object):

    def __init__(self,
                 layer,
                 cell_prev_prev,
                 cell_prev,
                 f_out,
                 type,
                 cell_id,
                 is_training,
                 activation='relu',
                 normalization='IN',
                 L2_weight=0.001,
                 multiplier=1.0,
                 summary_val=None,
                 summary_val_2=None,
                 summary_vis=None,
                 summary_vis_2=None):

        self.stride = 1
        self.cell_id = cell_id
        self.layer = layer
        self.weight_lenght = []
        self.alphas = []
        self.weights = []
        self.weights_eval = []

        if type is 'R':
            self.stride = 2
            self.layer = 1

        self.init_weights(self.layer, len(cell_prev.OPS), cell_id, type)

        if cell_prev_prev.features.shape[1] != cell_prev.features.shape[1]:
            node_prev_prev = Node(cell_prev_prev.features,
                                  f_out=f_out,
                                  stride=2,
                                  func_name='skip',
                                  cell_id=cell_id,
                                  layer=0,
                                  type=type,
                                  activation=activation,
                                  normalization=normalization,
                                  L2_weight=L2_weight,
                                  prev_nodes=[cell_prev_prev],
                                  is_training=is_training,
                                  name='H-2')
        else:
            node_prev_prev = Node(cell_prev_prev.features,
                                  f_out=f_out,
                                  stride=1,
                                  func_name='conv1x1',
                                  cell_id=cell_id,
                                  layer=0,
                                  type=type,
                                  activation=activation,
                                  normalization=normalization,
                                  L2_weight=L2_weight,
                                  prev_nodes=[cell_prev_prev],
                                  is_training=is_training,
                                  name='H-2')

        node_prev = Node(cell_prev.features,
                         f_out=f_out,
                         stride=1,
                         func_name='conv1x1',
                         cell_id=cell_id,
                         layer=0,
                         type=type,
                         activation=activation,
                         normalization=normalization,
                         L2_weight=L2_weight,
                         prev_nodes=[cell_prev],
                         is_training=is_training,
                         name='H-1')

        self.nodes = [[node_prev_prev, node_prev]]

        for layer in range(len(self.weights)):
            self.weights[layer] = tf.nn.softmax(self.weights[layer])

            std = reduce_std(self.weights[layer]) * multiplier
            var, _ = tf.nn.top_k(self.weights[layer], k=1)

            self.weights_eval[layer] = tf.where(self.weights[layer] >= var[0] - std,
                                              #self.weights[layer] >= 1.0 / int(self.weights[layer].shape[0]),
                                              tf.ones_like(self.weights[layer]),
                                              tf.zeros_like(self.weights[layer]))

            self.make_summary(self.weights[layer], cell_id, layer, summary_val, summary_vis, name='')
            self.make_summary(self.weights_eval[layer], cell_id, layer, summary_val_2, summary_vis_2, name='_eval')

        for layer in range(self.layer):
            net = 0
            prev_nodes = []
            weight_idx = 0
            for list_idx in range(len(self.nodes)):
                for nodes in range(len(self.nodes[list_idx])):
                    net += self.weights[list_idx][weight_idx] * self.nodes[list_idx][nodes].features
                    prev_nodes.append(self.nodes[list_idx][nodes])
                    weight_idx += 1

            new_nodes = []
            for func_name in cell_prev.OPS.keys():
                new_node = Node(net,
                                f_out=f_out,
                                stride=self.stride,
                                func_name=func_name,
                                cell_id=cell_id,
                                layer=layer + 1,
                                type=type,
                                activation=activation,
                                normalization=normalization,
                                L2_weight=L2_weight,
                                prev_weights=self.weights[layer],
                                prev_weights_eval=self.weights_eval[layer],
                                prev_nodes=prev_nodes,
                                is_training=is_training)
                new_nodes.append(new_node)

            self.nodes.append(new_nodes)

        net = []
        prev_nodes = []
        weight_idx = 0
        for list_idx in range(len(self.nodes)):
            for nodes in range(len(self.nodes[list_idx])):
                if self.nodes[list_idx][nodes].features.shape[1] == self.nodes[-1][-1].features.shape[1]:
                    net.append(self.weights[-1][weight_idx] * self.nodes[list_idx][nodes].features)
                    prev_nodes.append(self.nodes[list_idx][nodes])
                    weight_idx += 1

        net = tf.concat(net, axis=-1)

        new_node = Node(net,
                        f_out=f_out,
                        stride=1,
                        func_name='skip',
                        cell_id=cell_id,
                        layer=len(self.nodes),
                        type=type,
                        activation=activation,
                        normalization=normalization,
                        L2_weight=L2_weight,
                        prev_weights=self.weights[-1],
                        prev_weights_eval=self.weights_eval[-1],
                        prev_nodes=prev_nodes,
                        is_training=is_training,
                        name='concat')
        self.nodes.append([new_node])

    def init_weights(self, level, op_count, cell_id, type):

        with tf.variable_scope('cell_{}'.format(cell_id)):
            for n in range(level+1):
                weight_count = 2 + n * op_count
                if type is 'R' and n > 0:
                    weight_count -= 2
                self.weight_lenght.append(weight_count)

                weights = tf.get_variable(name='weight_{}_{}'.format(level, weight_count),
                                          shape=[weight_count],
                                          initializer=tf.random_normal_initializer(stddev=0.001,
                                                                                   mean=1.0 / weight_count),
                                          regularizer=tf.contrib.layers.l2_regularizer(scale=1e-3),
                                          trainable=True)

                self.weights.append(weights)
                self.weights_eval.append(weights)

    def make_summary(self, weights, cell_id, layer, summary_val, summary_vis, name=''):
        weight = tf.expand_dims(weights, axis=1)
        weight = tf.expand_dims(weight, axis=0)
        weight = tf.expand_dims(weight, axis=0)
        summary_vis.append(tf.summary.image("weight_C{}_L{}{}".format(cell_id, layer, name), weight))
        summary_val.append(tf.summary.text("weights_C{}_L{}{}".format(cell_id, layer, name), tf.as_string(weights)))