from tfcore.core.layer import *
from tfcore.utilities.params_serializer import ParamsSerializer


class Node_Params(ParamsSerializer):
    """
    Parameter class for ExampleModel
    """

    def __init__(self,
                 f_out=32,
                 stride=1,
                 func_name='none',
                 cell_id=0,
                 layer=0,
                 type='N',
                 pre_activation=True,
                 prev_nodes=[],
                 appendix=''):

        self.f_out = f_out
        self.stride = stride
        self.func_name = func_name
        self.cell_id = cell_id
        self.layer = layer
        self.type = type
        self.pre_activation = pre_activation
        self.prev_node_names = []
        if appendix == '':
            self.name_in_graph = 'C{}L{}_{}'.format(self.cell_id, self.layer, self.func_name)
        else:
            self.name_in_graph = 'C{}L{}_{}_{}'.format(self.cell_id, self.layer, self.func_name, appendix)
        #self.prev_node_names = [prev_node.params.name_in_graph for prev_node in prev_nodes]

    def set_params(self, params):
        self.__dict__.update(params)


class Node(object):

    def __init__(self,
                 net,
                 is_training,
                 f_out=32,
                 stride=1,
                 func_name='none',
                 cell_id=0,
                 layer=0,
                 type='N',
                 activation='none',
                 normalization='IN',
                 pre_activation=True,
                 prev_weights=tf.constant([1.0]),
                 prev_nodes=[],
                 name='',
                 params=None):

        self.is_training = is_training
        self.prev_weights = prev_weights
        self.prev_nodes = prev_nodes
        self.active = False
        self.normalization = normalization
        self.activation = activation

        self.params = params
        if self.params is None:
            self.params = Node_Params(f_out=f_out,
                                      stride=stride,
                                      func_name=func_name,
                                      cell_id=cell_id,
                                      layer=layer,
                                      type=type,
                                      pre_activation=pre_activation,
                                      prev_nodes=prev_nodes,
                                      appendix=name)

        self.features = self.get_function(net,
                                          f_out=self.params.f_out,
                                          stride=self.params.stride,
                                          func_name=self.params.func_name,
                                          name=self.params.name_in_graph)
        self.width = self.features.shape[1]
        self.height = self.features.shape[2]
        self.channel = self.features.shape[3]
        self.name = self.params.name_in_graph + \
                    ' {}x{}x{}'.format(self.features.shape[1], self.features.shape[2], self.features.shape[3])

    def get_function(self, net, f_out, stride, func_name, name):
        self.OPS = {'none': lambda net, f_out, stride, name: self.Zero(net=net, stride=stride, name=name),
                    'max_pool': lambda net, f_out, stride, name: self.max_pool(net=net, k_size=3, stride=stride,
                                                                               name=name),
                    'avg_pool': lambda net, f_out, stride, name: self.avg_pool(net=net, k_size=3, stride=stride,
                                                                               name=name),
                    'skip': lambda net, f_out, stride, name: self.factorize(net=net, f_out=f_out, name=name)
                    if stride > 1 else self.Identity(net=net),
                    'conv1x1': lambda net, f_out, stride, name: self.conv(net=net, k_size=1, f_out=f_out, stride=stride,
                                                                          name=name),
                    'conv3x3': lambda net, f_out, stride, name: self.conv(net=net, k_size=3, f_out=f_out,
                                                                          stride=stride,
                                                                          name=name),
                    'conv5x5': lambda net, f_out, stride, name: self.conv(net=net, k_size=5, f_out=f_out,
                                                                          stride=stride,
                                                                          name=name),
                    }

        return self.OPS[func_name](net, f_out, stride, name)

    def Zero(self, net, stride, name):
        if stride > 1:
            net = max_pool(net, stride=stride, name=name)
        return tf.multiply(net, 0)

    def Identity(self, net):
        return net

    def conv(self, net, k_size, f_out, stride, name):
        return conv2d(net,
                      k_size=k_size,
                      f_out=f_out,
                      stride=stride,
                      activation=self.activation,
                      normalization=self.normalization,
                      use_pre_activation=self.params.pre_activation,
                      is_training=self.is_training,
                      name=name)

    def factorize(self, net, f_out, name):
        return conv2d(net,
                      k_size=2,
                      f_out=f_out,
                      stride=2,
                      activation=self.activation,
                      normalization=self.normalization,
                      use_pre_activation=self.params.pre_activation,
                      is_training=self.is_training,
                      name=name)

    def max_pool(self, net, k_size, stride, name):
        with tf.variable_scope(name):
            net = max_pool(net, radius=k_size, stride=stride, name=name)
            norm_func = get_normalization(self.normalization, )
            return norm_func(net, training=self.is_training)

    def avg_pool(self, net, k_size, stride, name):
        with tf.variable_scope(name):
            net = avg_pool(net, radius=k_size, stride=stride, name=name)
            norm_func = get_normalization(self.normalization)
            return norm_func(net, training=self.is_training)
