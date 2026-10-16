"""Keras 2 symbols used by Rasa's TensorFlow layers.

TensorFlow 2.16+ installs Keras 3 as the top-level ``keras`` package.
``TF_USE_LEGACY_KERAS`` redirects ``tf.keras`` only. Rasa's layers still use
the Keras 2 API shipped in ``tf_keras``.
"""

from tf_keras import Model
from tf_keras.src.utils import tf_utils
from tf_keras.src.utils.control_flow_util import smart_cond

__all__ = ["Model", "smart_cond", "tf_utils"]
