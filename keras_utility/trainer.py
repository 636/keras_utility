
import logging
from typing import Callable, Generator, List, Union, Tuple

from keras import backend as K
from keras.models import Model

LOGGER = logging.getLogger(__name__)


# return: step, generator
DatasetLoader = Callable[[], Union[int, Generator]]


class KerasTrainer():

    logger = LOGGER.getChild('KerasTrainer')

    def __init__(self, worker=10, callbacks=[]):
        self.worker = worker
        self.callbacks = callbacks

    def train(self,
              model,
              train_loader: DatasetLoader,
              val_loader: DatasetLoader,
              epochs: int,
              callbacks=[]) -> Tuple[Model, dict]:

        _train_step, _train_loader = train_loader()

        self.logger.info('epoch: %s, train_step: %s',
                         epochs, _train_step)

        model.summary(print_fn=self.logger.info)

        _options = {
            'verbose': 1,
            'callbacks': self.callbacks + callbacks,
            'workers': self.worker
        }

        if val_loader:
            _val_step, _val_loader = val_loader()
            self.logger.info('val_step: %s', _val_step)
            _options.update({
                'validation_data': _val_loader,
                'validation_steps': _val_step,
            })

        self.logger.info('training start.')
        history = model.fit_generator(_train_loader,
                                      _train_step,
                                      epochs,
                                      **_options)
        self.logger.info('training end.')

        return model, history

    @classmethod
    def apply_multi_gpu_if_available(cls, model_builder: Callable, freeze: Callable = lambda x: x):
        """

        # return
            [is_multi: Bool, applyed multi gpu model, original model]
        """

        gpu_count = len(
            [x for x in [x.name for x in K.get_session().list_devices()] if 'gpu' in x.lower()])

        cls.logger.info('available gpu count: %s', gpu_count)

        is_multi = False
        if gpu_count > 1:
            cls.logger.info('apply multi gpu mode.')
            with tf.device('/cpu:0'):
                original_model = model_builder()
                original_model = freeze(original_model)

            model = multi_gpu_model(original_model)

            is_multi = True
        else:
            cls.logger.info(
                'not apply multi gpu. single gpu mode or cpu mode.')
            original_model = model_builder()
            original_model = freeze(original_model)

            model = original_model

        return is_multi, model, original_model
