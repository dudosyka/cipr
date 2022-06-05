import tensorflow as tf
import builtins
import pathlib

path = (pathlib.Path(__file__).parent.resolve()).joinpath('../../models/empty')

builtins.modelProto = tf.keras.models.load_model(path)
