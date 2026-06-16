#
# keras_compat.py
# Make the legacy CoNNECT model.h5 (saved with TF1.x / old Keras) loadable
# under Keras 3.
#
# Old HDF5 models embed a 'dtype' key inside initializer configs (e.g.
# {'class_name': 'GlorotUniform', 'config': {'seed': None, 'dtype': 'float32'}}).
# Keras 3 initializers' __init__ no longer accept 'dtype', so deserialization
# raises "GlorotUniform.__init__() got an unexpected keyword argument 'dtype'".
# Patch Initializer.from_config to drop the stale 'dtype' key.
#
# Import this module before calling keras.models.load_model().
#

import keras

_Initializer = keras.initializers.Initializer

if not getattr(_Initializer, "_connect_dtype_patch", False):
    def from_config(cls, config):
        config = dict(config)
        config.pop("dtype", None)
        return cls(**config)

    _Initializer.from_config = classmethod(from_config)
    _Initializer._connect_dtype_patch = True
