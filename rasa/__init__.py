import logging
import os

# TensorFlow 2.16+ defaults to Keras 3. Rasa's layers target the Keras 2 API
# shipped as tf-keras. This must be set before TensorFlow is imported.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from rasa import version, plugin  # noqa: F401
from rasa.api import run, train, test  # noqa: F401

# define the version before the other imports since these need it
__version__ = version.__version__


logging.getLogger(__name__).addHandler(logging.NullHandler())
