from darts.trainer import *
from tfcore.interfaces.IPipeline_Trainer import *
from tfcore.utilities.preprocessing import Preprocessing
import gflags
import os
import sys

class Pipeline_Params(IPipeline_Trainer_Params):
    """ Simple example for Pipeline_Params

    """

    def __init__(self,
                 data_dir_y,
                 data_dir_x,
                 validation_dir_x,
                 validation_dir_y,
                 output_dir,
                 convert=True,
                 epochs=25,
                 batch_size=16,
                 shuffle=True,
                 cache_size=1,
                 interp='bicubic'):
        super().__init__(data_dir_x=data_dir_x,
                         data_dir_y=data_dir_y,
                         validation_dir_x=validation_dir_x,
                         validation_dir_y=validation_dir_y,
                         output_dir=output_dir,
                         convert=convert,
                         epochs=epochs,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         cache_size=cache_size,
                         interp=interp)
        self.validation_dir_x = validation_dir_x


class Training_Pipeline(IPipeline_Trainer):
    """ Simple example of inherent from IPipeline and to create an class

    # Arguments
        trainer: Implementation of meta class ITrainer
        params: Implementation of meta class IPipeline_Params
        pre_processing: Implementation of class Preprocessing
    """

    def __init__(self, trainer, params, pre_processing):

        super().__init__(trainer, params, pre_processing)

    def get_element(self, idx):

        try:
            img_x = imageio.imread(self.files_x[idx])
            img_y = self.files_x[idx].count('good')
        except FileNotFoundError:
            raise FileNotFoundError(' [!] File not found of data-set x')

        if self.pre_processing is not None:
            img_x, _ = self.pre_processing.run(img_x, None)

        return img_x, np.asarray(img_y)

    def set_validation(self):
        batch_val_x = []
        batch_val_y = []

        pre_processing = Preprocessing()
        pre_processing.add_function_x(Preprocessing.Central_Crop(size=(960, 960)).function)
        pre_processing.add_function_x(Preprocessing.Crop_by_Center(treshold=32, size=(224*2, 224*2)).function)
        pre_processing.add_function_x(Preprocessing.DownScale(factors=4).function)

        for idx in range(len(self.files_val_x)):
            try:
                img_x = imageio.imread(self.files_val_x[idx])
                img_x, _ = pre_processing.run(img_x, None)
                img_y = self.files_val_x[idx].count('good')

                batch_val_x.append(img_x)
                batch_val_y.append(img_y)

            except FileNotFoundError:
                raise FileNotFoundError(' [!] File not found of data-set x')

        try:
            self.trainer.set_validation_set(np.asarray(batch_val_x), np.asarray(batch_val_y))
        except Exception as err:
            print(' [!] Error in Trainer on set_validation():', err)
            raise


#   Flaks to configure from shell
flags = gflags.FLAGS
gflags.DEFINE_string("config_path", '', "Path for config files")
gflags.DEFINE_string("dataset", "../Data/", "Dataset path")
gflags.DEFINE_string("model_dir", "../pretrained_models/", "Dataset path")


def main():
    flags(sys.argv)

    #   Trainer_Params witch inherits from ITrainer_Params
    model_params = Trainer_Params(image_size=224/2,
                                  params_path=flags.config_path,
                                  model_path=flags.model_dir)
    #   Trainer witch inherits from ITrainer
    model_trainer = Trainer(model_params)

    #   Pre-processing Pipeline
    pre_processing = Preprocessing()
    pre_processing.add_function_x(Preprocessing.Central_Crop(size=(960,960)).function)
    pre_processing.add_function_x(Preprocessing.Crop_by_Center(treshold=32, size=(224*2, 224*2)).function)
    pre_processing.add_function_x(Preprocessing.DownScale(factors=4).function)

    #   Pipeline_Params witch inherits from IPipeline_Params
    pipeline_params = Pipeline_Params(data_dir_x=os.path.join(flags.dataset, 'train'),
                                      data_dir_y=None,
                                      validation_dir_x=os.path.join(flags.dataset, 'test'),
                                      validation_dir_y=None,
                                      batch_size=model_params.batch_size,
                                      epochs=model_params.epoch,
                                      convert=False,
                                      shuffle=True,
                                      output_dir=None)

    #   Pipeline witch inherits from IPipeline
    pipeline = Training_Pipeline(trainer=model_trainer, params=pipeline_params, pre_processing=pre_processing)

    pipeline.set_validation()

    #   Start Training
    pipeline.run()


if __name__ == "__main__":
    main()
