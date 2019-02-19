# -*- coding: utf-8 -*-

import logging
from datetime import datetime
from pathlib import Path
import yaml

from keras.models import Model

LOGGER = logging.getLogger(__name__)


def _timestamp_formatter(key):
    suffix = datetime.utcnow().strftime('%Y-%m/%d-%H%M-%S')
    return '{}-{}'.format(key, suffix)


class DirectoryModelStore:

    logger = LOGGER.getChild('DirectoryModelStore')

    def __init__(self, base_dir: Path, group_name: str, model_key_formatter=_timestamp_formatter):

        self.base_dir = base_dir
        self.group_name = group_name
        self.model_key_formatter = model_key_formatter

    def new_model_key(self, prefix='model'):
        return self.model_key_formatter(prefix)

    def save(self, model_key: str, model: Model):

        _save_dir = self.base_dir / self.group_name / model_key
        _save_dir.mkdir(parents=True, exist_ok=True)

        model.save(str(_save_dir / 'model-all.h5'))
        model.save_weights(str(_save_dir / 'model-weight.h5'))
        with (_save_dir / 'model.yaml').open('w', encoding='utf-8') as f:
            f.write(model.to_yaml())
