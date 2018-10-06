import tensorflow as tf
from CNN.node import Node


class Cell(object):

    def __init__(self,
                 layer,
                 cell_prev_prev,
                 cell_prev,
                 f_out,
                 type,
                 cell_id,
                 is_training,
                 is_eval,
                 activation='relu',
                 normalization='IN'):

        self.stride = 1
        self.cell_id = cell_id
        self.layer = layer
        self.weight_lenght = []
        self.alphas = []
        self.weights = []

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
                         prev_nodes=[cell_prev],
                         is_training=is_training,
                         name='H-1')

        self.nodes = [[node_prev_prev, node_prev]]

        for layer in range(len(self.weights)):
            self.weights[layer] = tf.nn.softmax(self.alphas[layer])

            self.weights[layer] = tf.cond(is_eval,
                                          lambda: tf.where(
                                              self.weights[layer] >= 1.0 / int(self.weights[layer].shape[0]),
                                              tf.ones_like(self.weights[layer]),
                                              tf.zeros_like(self.weights[layer])),
                                          lambda: self.weights[layer])

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
                                layer=layer,
                                type=type,
                                activation=activation,
                                normalization=normalization,
                                prev_weights=self.weights[layer],
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
                        prev_weights=self.weights[layer],
                        prev_nodes=prev_nodes,
                        is_training=is_training)
        self.nodes.append([new_node])

    def init_weights(self, level, op_count, cell_id, type):

        with tf.variable_scope('cell_{}'.format(cell_id)):
            for n in range(level+1):
                weight_count = 2 + n * op_count
                if type is 'R' and n > 0:
                    weight_count -= 2
                self.weight_lenght.append(weight_count)
                alphas = tf.get_variable(name='alpha_{}_{}'.format(level, weight_count),
                                         shape=[weight_count],
                                         initializer=tf.random_normal_initializer(stddev=0.001,
                                                                                  mean=1.0 / weight_count),
                                         trainable=True)

                weights = tf.get_variable(name='weight_{}_{}'.format(level, weight_count),
                                          shape=[weight_count],
                                          initializer=tf.random_normal_initializer(stddev=0.001,
                                                                                   mean=1.0 / weight_count),
                                          trainable=True)

                self.alphas.append(alphas)
                self.weights.append(weights)
