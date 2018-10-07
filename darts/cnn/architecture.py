import os
from graphviz import Digraph
from tfcore.interfaces.IModel import IModel, IModel_Params
from tfcore.core.layer import *
from tfcore.core.activations import *
from tfcore.core.loss import *

from CNN.node import Node, Node_Params
from CNN.cell import Cell


class Network_Params(IModel_Params):
    """
    Parameter class for ExampleModel
    """

    def __init__(self,
                 f_start=32,
                 activation='relu',
                 normalization='IN',
                 scope='Classifier',
                 name='Classifier'):
        super().__init__(scope=scope, name=name)

        self.f_start = f_start
        self.activation = activation
        self.normalization = normalization
        self.path = os.path.realpath(__file__)


class Network(IModel):
    """
    Example of a simple 3 layer generator model for super-resolution
    """

    def __init__(self, sess, params, global_steps, is_training, is_eval):
        """
        Init of Example Class

        # Arguments
            sess: Tensorflow-Session
            params: Instance of ExampleModel_Params
            global_steps: Globel steps for optimizer
            is_training: placeholder variable for switch between training end eval phase
        """
        super().__init__(sess, params, global_steps)
        self.model_name = self.params.name
        self.activation = get_activation(name='relu')
        self.normalization = get_normalization(self.params.normalization)
        self.is_training = is_training
        self.is_eval = is_eval

    def build_model(self, input, is_train=False, reuse=False):
        """
        Build model and create summary

        # Arguments
            input: Input-Tensor
            is_train: Bool
            reuse: Bool

        # Return
            Tensor of dimension 4D
        """
        self.reuse = reuse
        super().build_model(input, is_train, reuse)

        return self.probs

    def model(self, net, is_train=False, reuse=False):
        """
        Create generator model

        # Arguments
            input: Input-Tensor
            is_train: Bool
            reuse: Bool

        # Return
            Tensor of dimension 4D
        """

        f_out = self.params.f_start
        with tf.variable_scope(self.params.scope, reuse=tf.AUTO_REUSE):

            cell_prev_prev = Node(net=net,
                                  f_out=f_out,
                                  stride=1,
                                  func_name='conv3x3',
                                  cell_id=0,
                                  layer=0,
                                  type='N',
                                  activation=self.params.activation,
                                  normalization=self.params.normalization,
                                  is_training=self.is_training,
                                  name='input-2')

            cell_prev = Node(net=net,
                             f_out=f_out,
                             stride=1,
                             func_name='conv3x3',
                             cell_id=0,
                             layer=0,
                             type='N',
                             activation=self.params.activation,
                             normalization=self.params.normalization,
                             is_training=self.is_training,
                             name='input-1')

            cell_types = ['N', 'R', 'N', 'R', 'N', 'R']
            self.nodes = [cell_prev_prev, cell_prev]
            for cell_id in range(len(cell_types)):
                if cell_types[cell_id] == 'R':
                    f_out *= 2

                cell = Cell(layer=3,
                            cell_prev_prev=self.nodes[-2],
                            cell_prev=self.nodes[-1],
                            f_out=f_out,
                            type=cell_types[cell_id],
                            cell_id=cell_id,
                            activation=self.params.activation,
                            normalization=self.params.normalization,
                            is_training=self.is_training,
                            is_eval=self.is_eval)
                self.nodes.append(cell.nodes[-1][-1])
                self.make_summary(cell)

            net = conv2d(self.nodes[-1].features,
                         k_size=1,
                         f_out=2,
                         stride=1,
                         activation=self.params.activation,
                         normalization=self.params.normalization,
                         use_pre_activation=True,
                         is_training=self.is_training,
                         name='conv_GAP')
            net = avg_pool(net, radius=net.shape[1], stride=1, padding='VALID', name='GAP')
            net = tf.reduce_mean(net, axis=[1, 2])

            self.logits = net
            self.probs = tf.nn.softmax(net)

        print(' [*] DARTS loaded...')
        return self.logits

    def make_summary(self, cell):
        for weight in cell.weights:
            weight = tf.expand_dims(weight, axis=1)
            weight = tf.expand_dims(weight, axis=0)
            weight = tf.expand_dims(weight, axis=0)
            self.summary_vis.append(tf.summary.image("test_weight_{}".format(cell.cell_id), weight))
            self.summary_vis_2.append(tf.summary.image("eval_weight_{}".format(cell.cell_id), weight))

    def loss(self, Y, normalize=False, name='MSE'):

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=self.logits))

        # Weight decay regularizer
        regularizer_L2 = 0.001 * tf.add_n(tf.get_collection('losses'))

        # total loss by the mean of cross entropy loss and the weighted regularizer
        self.total_loss = tf.reduce_mean(loss + regularizer_L2)

        # Accuracy for train and test set
        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(self.probs, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Summarys for tensorboard
        self.summary.append(tf.summary.scalar("accuracy_train", accuracy))
        self.summary_val.append(tf.summary.scalar("accuracy_test", accuracy))
        self.summary_val_2.append(tf.summary.scalar("accuracy_eval", accuracy))

        self.summary.append(tf.summary.scalar("cross_entropy_train", loss))
        self.summary_val.append(tf.summary.scalar("cross_entropy_test", loss))
        self.summary_val_2.append(tf.summary.scalar("cross_entropy_eval", loss))
        self.summary.append(tf.summary.scalar("Learning rate", self.learning_rate))

        self.summary.append(tf.summary.scalar("total_loss_train", self.total_loss))
        self.summary_val.append(tf.summary.scalar("total_loss_test", self.total_loss))
        self.summary_val_2.append(tf.summary.scalar("total_loss_eval", self.total_loss))

        return self.total_loss

    def make_graph(self, path, filename):
        last_node = [[self.nodes[-1]]]
        found = True
        while found:
            for node in last_node[0]:
                try:
                    if len(node.prev_nodes) > 0:

                        weigths = tf.where(node.prev_weights >= 1.0 / len(node.prev_nodes),
                                               tf.ones_like(node.prev_weights),
                                               tf.zeros_like(node.prev_weights))
                        index = self.sess.run([weigths], feed_dict={self.is_eval: True})[0]

                        nodes = []
                        for idx in range(len(index)):
                            if index[idx] == 1.0:
                                node.prev_nodes[idx].active = True
                                nodes.append(node.prev_nodes[idx])
                            else:
                                node.prev_nodes[idx].active = False
                        last_node.insert(0, nodes)
                    else:
                        found = False
                except:
                    print ('')

        graph = Digraph('G', filename='hello.gv', format='png')
        last_node = [[self.nodes[-1]]]
        found = True
        edges = []
        while found:
            nodes = []
            for node in last_node[0]:
                if len(node.prev_nodes) == 0:
                    found = False
                    break

                for prev_node in node.prev_nodes:
                    if prev_node.active and node.active:
                        if [prev_node, node] not in edges:
                            graph.edge(prev_node.name, node.name)
                            edges.append([prev_node, node])
                        if prev_node not in nodes:
                            nodes.append(prev_node)
            last_node.insert(0, nodes)
        graph.render(filename=filename, directory=path)

        print (' [*] Graph drawed')

