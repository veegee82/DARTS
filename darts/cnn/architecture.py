import os
from graphviz import Digraph
from tfcore.interfaces.IModel import IModel, IModel_Params
from tfcore.core.layer import *
from tfcore.core.activations import *
from tfcore.core.loss import *
from tfcore.utilities.params_serializer import ParamsSerializer
from tfcore.utilities.utils import reduce_std
from cnn.node import Node, Node_Params
from cnn.cell import Cell


class Architecture(ParamsSerializer):

    def __init__(self, node_params=[]):

        self.node_params = node_params

    def load(self, path):
        """ Load Parameter

        # Arguments
            path: Path of json-file
        # Return
            Parameter class
        """
        super().load(os.path.join(path))
        for list in range(len(self.node_params)):
            for params in range(len(self.node_params[list])):
                node_params = Node_Params()
                node_params.set_params(self.node_params[list][params])
                self.node_params[list][params] = node_params

    def save(self, path):
        """ Save parameter as json-file

        # Arguments
            path: Path to save
        """
        if not os.path.exists(path):
            os.makedirs(path)
        super().save(os.path.join(path))
        return

class Network_Params(IModel_Params):
    """
    Parameter class for ExampleModel
    """

    def __init__(self,
                 f_start=32,
                 activation='relu',
                 normalization='IN',
                 multiplier=1.750,
                 load_model=False,
                 L2_weight=3e-4,
                 model_name='',
                 scope='Classifier',
                 name='Classifier'):
        super().__init__(scope=scope, name=name)

        self.load_model = load_model
        self.model_name = model_name
        self.f_start = f_start
        self.activation = activation
        self.normalization = normalization
        self.L2_weight = L2_weight
        self.path = os.path.realpath(__file__)
        self.multiplier = multiplier

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
        if not self.params.load_model:
            super().build_model(input, is_train, reuse)
        else:
            self.load_model(input, self.params.model_name)

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
                                  normalization=self.params.normalization,
                                  L2_weight=self.params.L2_weight,
                                  is_training=self.is_training,
                                  name='input-2')

            cell_prev = Node(net=net,
                             f_out=f_out,
                             stride=1,
                             func_name='conv3x3',
                             cell_id=0,
                             layer=0,
                             type='N',
                             normalization=self.params.normalization,
                             L2_weight=self.params.L2_weight,
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
                            L2_weight=self.params.L2_weight,
                            is_training=self.is_training,
                            multiplier=self.params.multiplier,
                            summary_val=self.summary_val,
                            summary_val_2=self.summary_val_2,
                            summary_vis=self.summary_vis,
                            summary_vis_2=self.summary_vis_2)
                self.nodes.append(cell.nodes[-1][-1])

            net = conv2d(self.nodes[-1].features,
                         k_size=1,
                         f_out=2,
                         stride=1,
                         activation=self.params.activation,
                         normalization=self.params.normalization,
                         use_pre_activation=True,
                         L2_weight=self.params.L2_weight,
                         is_training=self.is_training,
                         name='conv_GAP')
            net = avg_pool(net, radius=net.shape[1], stride=1, padding='VALID', name='GAP')
            net = tf.reduce_mean(net, axis=[1, 2])

            self.logits = net
            self.probs = tf.nn.softmax(net)

        print(' [*] DARTS loaded...')
        return self.logits

    def load_model(self, net, model_name):

        architecture = Architecture()
        architecture.load(model_name)

        def find_params(name, params_list):
            for list in params_list:
                for node_params in list:
                    if node_params.name_in_graph == name:
                        return node_params
            return None

        def find_node(node_name, node_list):
            for node in node_list:
                if node_name == node.params.name_in_graph:
                    return node
            return None

        def create_prev_node(node_params, node_list, params_list):
            concat = []
            for prev_node_name in node_params.prev_node_names:
                node = find_node(prev_node_name, node_list)
                prev_params = find_params(prev_node_name, params_list)

                if node is None and prev_params is not None:
                    node = create_prev_node(prev_params, node_list, params_list)
                if node is not None:
                    concat.append(node.features)

            if 'concat' in node_params.name_in_graph:
                net = tf.concat(concat, axis=-1)
            else:
                net = tf.add_n(concat) / len(concat)

            new_node = Node(net=net,
                            activation=self.params.activation,
                            normalization=self.params.normalization,
                            params=node_params,
                            L2_weight=self.params.L2_weight,
                            is_training=self.is_training)
            node_list.append(new_node)
            return new_node

        f_out = self.params.f_start
        with tf.variable_scope(self.params.scope, reuse=tf.AUTO_REUSE):
            cell_prev_prev = Node(net=net,
                                  f_out=f_out,
                                  stride=1,
                                  func_name='conv3x3',
                                  cell_id=0,
                                  layer=0,
                                  type='N',
                                  normalization=self.params.normalization,
                                  L2_weight=self.params.L2_weight,
                                  is_training=self.is_training,
                                  name='input-2')

            cell_prev = Node(net=net,
                             f_out=f_out,
                             stride=1,
                             func_name='conv3x3',
                             cell_id=0,
                             layer=0,
                             type='N',
                             normalization=self.params.normalization,
                             L2_weight=self.params.L2_weight,
                             is_training=self.is_training,
                             name='input-1')

            node = create_prev_node(architecture.node_params[0][0],
                                    [cell_prev_prev, cell_prev],
                                    architecture.node_params)

            net = conv2d(node.features,
                         k_size=1,
                         f_out=2,
                         stride=1,
                         activation=self.params.activation,
                         normalization=self.params.normalization,
                         L2_weight=self.params.L2_weight,
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
            self.summary_vis.append(tf.summary.image("weight_{}".format(cell.cell_id), weight))
            self.summary_vis_2.append(tf.summary.image("weight_{}_eval".format(cell.cell_id), weight))

    def loss(self, Y, normalize=False, name='MSE'):

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=self.logits))

        # Weight decay regularizer
        l2_loss = tf.losses.get_regularization_loss()

        # total loss by the mean of cross entropy loss and the weighted regularizer
        self.total_loss = loss + l2_loss

        # Accuracy for train and test set
        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(self.probs, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Summarys for tensorboard
        self.summary.append(tf.summary.scalar("accuracy_train", accuracy))
        self.summary_val.append(tf.summary.scalar("accuracy_test", accuracy))

        self.summary.append(tf.summary.scalar("cross_entropy_train", loss))
        self.summary_val.append(tf.summary.scalar("cross_entropy_test", loss))

        self.summary.append(tf.summary.scalar("total_loss_train", self.total_loss))
        self.summary_val.append(tf.summary.scalar("total_loss_test", self.total_loss))

        self.summary_val.append(tf.summary.scalar("l2_loss", l2_loss))
        return self.total_loss

    def make_graph(self, path, filename):
        self.nodes[-1].active = True
        last_node = [[self.nodes[-1]]]
        found = True
        while found:
            nodes = []
            for node in last_node[0]:

                if len(node.prev_nodes) > 0:
                    index = self.sess.run([node.prev_weights_eval], feed_dict={self.is_eval: True})[0]
                    node.params.prev_node_names = []
                    for idx in range(len(index)):
                        if index[idx] == 1.0:
                            node.prev_nodes[idx].active = True
                            if node.prev_nodes[idx].params.name_in_graph not in node.params.prev_node_names:
                                node.params.prev_node_names.append(node.prev_nodes[idx].params.name_in_graph)
                            if node.prev_nodes[idx] not in nodes:
                                nodes.append(node.prev_nodes[idx])
                        else:
                            node.prev_nodes[idx].active = False
            if len(nodes) > 0:
                last_node.insert(0, nodes)
            else:
                found = False

        graph = Digraph('G', filename='hello.gv', format='png')
        last_node = [[self.nodes[-1]]]
        found = True
        edges = []
        architecture_params = [[self.nodes[-1].params.__dict__]]

        def check_if_active(nodes):
            for node in nodes:
                if node.active:
                    return True
            return False

        while found:
            nodes = []
            params = []
            for node in last_node[0]:
                if len(node.prev_nodes) == 0:
                    found = False
                    break

                for prev_node in node.prev_nodes:
                    if prev_node.active and node.active and \
                            (check_if_active(prev_node.prev_nodes) or len(prev_node.prev_nodes) == 0):
                        if [prev_node, node] not in edges:
                            graph.edge(prev_node.name, node.name)
                            edges.append([prev_node, node])
                            if prev_node not in nodes:
                                nodes.append(prev_node)
                                params.append(prev_node.params.__dict__)
            if len(nodes) > 0:
                last_node.insert(0, nodes)
                architecture_params.append(params)
            else:
                found = False
        graph.render(filename=filename, directory=path)

        architecture = Architecture(node_params=architecture_params)
        architecture.save(path=os.path.join(path, filename))

        print (' [*] Graph drawed')

