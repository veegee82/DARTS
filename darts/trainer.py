import time
from tfcore.interfaces.ITraining import *
from CNN.architecture import Network, Network_Params
from tfcore.utilities.preprocessing import *
from tfcore.utilities.files import get_filename


class Trainer_Params(ITrainer_Params):
    """
    Parameter-Class for Example-Trainer
    """

    def __init__(self,
                 image_size,
                 params_path='',
                 model_path='',
                 load=True,
                 ):
        """ Constructor

        # Arguments
            image_size: Image size (int)
            params_path: Parameter-Path for loading and saving (str)
            model_path: Model path for pretrained model
            load: Load parameter (boolean)
        """
        super().__init__()

        self.image_size = image_size
        self.epoch = 2000
        self.batch_size = 16
        self.decay = 0.99
        self.step_decay = 100
        self.beta1 = 0.9
        self.learning_rate_G = 0.0005
        self.use_tensorboard = True
        self.gpus = [0]
        self.input_norm = 'tanh'
        self.normalization_G = 'IN'
        self.cyclic_LR = False

        self.use_pretrained_generator = True
        self.pretrained_generator_dir = model_path

        self.experiment_name = "Experiment_DARTS"
        self.checkpoint_restore_dir = ''
        self.load_checkpoint = False

        self.use_validation_set = True
        self.evals_per_iteration = 25
        self.save_checkpoint = False

        if params_path is not '':
            if load:
                if self.load(params_path):
                    self.save(params_path)
            else:
                self.save(params_path)

        self.root_dir = "../../Results/"

    def load(self, path):
        """ Load Parameter

        # Arguments
            path: Path of json-file
        # Return
            Parameter class
        """
        return super().load(os.path.join(path, "Trainer_Params"))

    def save(self, path):
        """ Save parameter as json-file

        # Arguments
            path: Path to save
        """
        if not os.path.exists(path):
            os.makedirs(path)
        super().save(os.path.join(path, "Trainer_Params"))
        return


class Trainer(ITrainer):
    """ A example class to train a generator neural-network
    """

    def __init__(self, trainer_params):
        """
        # Arguments
            trainer_params: Parameter from class Example_Trainer_Params
        """

        #   Initialize the abstract Class ITrainer
        super().__init__(trainer_params)

        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
        self.is_eval = tf.placeholder(tf.bool, shape=[], name='is_eval')

        #   Load Model
        network_params = Network_Params(activation='relu',
                                        normalization=self.params.normalization_G)

        network_params.decay = self.params.decay
        network_params.step_decay = self.params.step_decay
        network_params.beta1 = self.params.beta1
        network_params.learning_rate = self.params.learning_rate_G

        self.network = Network(self.sess, network_params, self.global_step, self.is_training, self.is_eval)

        #   Create the directorys for logs, checkpoints and samples
        self.prepare_directorys()
        #   Save the hole dl_core library as zip

        #   Placeholder for input x
        self.all_X = tf.placeholder(tf.float32,
                                    [None,
                                     self.params.image_size,
                                     self.params.image_size,
                                     1],
                                    name='all_X')

        #   Placeholder for ground-truth Y
        self.all_Y = tf.placeholder(tf.float32,
                                    [None, 2],
                                    name='all_Y')

        #   Build Pipeline
        self.build_pipeline()

        #   Initialize all variables
        tf.global_variables_initializer().run(session=self.sess)
        self.sess.run(tf.local_variables_initializer())
        print(' [*] All variables initialized...')

        self.saver = tf.train.Saver()

        #   Load pre-trained model
        if self.params.use_pretrained_generator or self.params.new is False:
            self.network.load(self.params.pretrained_generator_dir)

        #   Load checkpoint
        if self.params.load_checkpoint:
            load(self.sess, self.params.checkpoint_restore_dir)

        return

    def prepare_directorys(self):
        """ Prepare the directorys for logs, samples and checkpoints
        """
        self.model_dir = "%s__%s" % (
            self.network.params.name, self.batch_size)
        self.checkpoint_dir = os.path.join(self.params.root_dir,
                                           self.params.experiment_name)
        self.log_dir = os.path.join(self.checkpoint_dir, 'logs')
        self.architectur_dir = os.path.join(self.checkpoint_dir, 'architectures')

        if self.params.new:
            if os.path.exists(self.checkpoint_dir):
                shutil.rmtree(self.checkpoint_dir)
            if os.path.exists(self.log_dir):
                shutil.rmtree(self.log_dir)
            if os.path.exists(self.architectur_dir):
                shutil.rmtree(self.architectur_dir)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.architectur_dir):
            os.makedirs(self.architectur_dir)

        #   Save the Trainer_Params as json
        self.params.save(self.checkpoint_dir)

    def set_validation_set(self, batch_valid_X, batch_valid_Y):
        """ Set the validation set for x and Y which same batch-size like training-examples

        # Arguments
            batch_valid_X: samples for x
            batch_valid_Y: samples for Y
        """

        if batch_valid_X.ndim < 4:
            batch_valid_X = np.expand_dims(batch_valid_X, axis=-1)

        self.image_val = normalize(batch_valid_X, normalization_type=self.params.input_norm)
        self.label_val = np.eye(2)[np.array([batch_valid_Y]).reshape(-1)]

        print(' [*] Image_X ' + str(self.image_val.shape))
        print(' [*] Label_Y ' + str(self.label_val.shape))

    def validate(self, epoch, iteration, idx):
        """ Validate the validation-set

        # Arguments
            epoch:   Current epoch
            iteration: Current interation
            idx: Index in current epoch
        """

        #   Validate Samples
        g_loss_val, g_summery, g_summery_vis = self.sess.run([self.network.total_loss,
                                                              self.summary_val,
                                                              self.summary_vis],
                                                             feed_dict={self.all_X: self.image_val,
                                                                        self.all_Y: self.label_val,
                                                                        self.epoch: epoch,
                                                                        self.network.learning_rate: self.params.learning_rate_G,
                                                                        self.is_training: False,
                                                                        self.is_eval: False})

        g_loss_val_2, g_summery_2, g_summery_vis_2 = self.sess.run([self.network.total_loss,
                                                                    self.summary_val_2,
                                                                    self.summary_vis_2],
                                                                   feed_dict={self.all_X: self.image_val,
                                                                              self.all_Y: self.label_val,
                                                                              self.epoch: epoch,
                                                                              self.network.learning_rate: self.params.learning_rate_G,
                                                                              self.is_training: False,
                                                                              self.is_eval: True})

        self.writer.add_summary(g_summery, iteration)
        self.writer.add_summary(g_summery_vis, iteration)

        self.writer.add_summary(g_summery_2, iteration)
        self.writer.add_summary(g_summery_vis_2, iteration)

        if iteration == 0:
            g_loss_val, g_summery = self.sess.run([
                self.network.total_loss,
                self.summary_vis_one],
                feed_dict={self.all_X: self.image_val,
                           self.all_Y: self.label_val,
                           self.epoch: epoch,
                           self.is_training: False,
                           self.is_eval: False})

            self.writer.add_summary(g_summery, iteration)

        if np.mod(epoch + 1, 5) == 0:
            self.network.make_graph(path=self.architectur_dir,
                                    filename='Epoch_{}'.format(get_filename(epoch, extension='')))

        print("[Sample] g_loss: {}, eval_loss: {}".format(g_loss_val, g_loss_val_2))

    def make_summarys(self, gradient_list=[]):
        """ Calculate some metrics and add it to the summery for the validation-set

        # Arguments
            gradient_list: Gradients to store in log-file as histogram
        """

        self.models.append(self.network)

        super().make_summarys(gradient_list)

    def set_losses(self, Y):

        self.g_loss = self.network.loss(Y)

    def build_model(self, tower_id):
        """ Build models for U-Net

        Paper:

        # Arguments
            tower_id: Tower-ID
        # Return
            List of all existing models witch should trained
        """

        #   Split the total batch by the gpu-count
        X = self.all_X[tower_id * self.batch_size:(tower_id + 1) * self.batch_size, :]
        Y = self.all_Y[tower_id * self.batch_size:(tower_id + 1) * self.batch_size, :]

        #   Create generator model
        self.network.build_model(X)

        self.set_losses(Y)

        #   Append all models with should be optimized
        model_list = []
        model_list.append(self.network)

        t_vars = tf.trainable_variables()
        self.g_vars = [var for var in t_vars if not ('alpha' in var.name or 'weight' in var.name)]
        self.alpha_vars = [var for var in t_vars if ('alpha' in var.name or 'weight' in var.name)]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.g_optim = tf.train.AdamOptimizer(self.network.learning_rate, beta1=self.params.beta1).minimize(
                self.network.total_loss, var_list=self.g_vars)
            self.alpha_optim = tf.train.AdamOptimizer(0.005, beta1=self.params.beta1).minimize(self.network.total_loss,
                                                                                               var_list=self.alpha_vars)

        return []  # model_list

    def train_online(self, batch_X, batch_Y, epoch=0, counter=1, idx=0, batch_total=0):
        """ Training, validating and saving of the generator model

        # Arguments
            batch_X: Training-Examples for input x
            batch_Y: Training-Examples for ground-truth Y
            epoch: Current epoch
            counter: Current iteration
            idx: Current batch
            batch_total: Total batch size
        """

        if batch_X.shape[0] != self.params.batch_size or batch_Y.shape[0] != self.params.batch_size:
            print(' [!] Wrong batch size')
            return

        start_time = time.time()

        #   Cyclic learning rate
        if self.params.cyclic_LR:
            self.params.learning_rate_G = self.network.crl.get_learning_rate(counter)

        #   Normalize input images between -1 and 1 + data augumentaion
        pre_processing = Preprocessing()
        pre_processing.add_function_x(Preprocessing.Rotate(steps=1).function)
        pre_processing.add_function_x(Preprocessing.Flip().function)
        for i in range(self.params.batch_size):
            batch_X[i], _ = pre_processing.run(batch_X[i], None)

        batch_X = np.asarray(batch_X)
        if batch_X.ndim < 4:
            batch_X = np.expand_dims(batch_X, axis=-1)

        self.images_train = normalize(batch_X, normalization_type=self.params.input_norm)
        self.label_train = np.eye(2)[np.array([batch_Y]).reshape(-1)]

        #   Validate after N iterations
        if epoch < 2:
            if np.mod(counter, self.params.evals_per_iteration) == 0:
                self.validate(epoch, counter, idx)
        else:
            if np.mod(counter, batch_total) == 0:
                self.validate(epoch, counter, idx)

        # Optimize Classifier

        feed_dict = {self.all_X: self.images_train,
                     self.all_Y: self.label_train,
                     self.epoch: epoch,
                     self.network.learning_rate: self.params.learning_rate_G,
                     self.is_training: True,
                     self.is_eval: False}

        _, g_loss, summary_G = self.sess.run([self.g_optim, self.network.total_loss, self.summary],
                                             feed_dict=feed_dict)

        self.writer.add_summary(summary_G, counter)

        feed_dict = {self.all_X: self.image_val,
                     self.all_Y: self.label_val,
                     self.epoch: epoch,
                     self.network.learning_rate: self.params.learning_rate_G,
                     self.is_training: True,
                     self.is_eval: False}

        _, alpha_loss, summary_G = self.sess.run([self.alpha_optim, self.network.total_loss, self.summary],
                                                 feed_dict=feed_dict)

        self.writer.add_summary(summary_G, counter)

        print("Train UNET: Epoch: [%2d] [%4d/%4d] time: %4.4f, g_loss: %.8f, alpha_loss: %.8f"
              % (epoch, idx, batch_total, time.time() - start_time, g_loss, alpha_loss))

        #   Save model and checkpoint
        if np.mod(counter + 1, 50) == 0:
            self.network.save(self.sess, self.checkpoint_dir, self.global_step)
