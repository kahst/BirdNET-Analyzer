import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
from tensorflow import lite as tflite
from tensorflow import keras

import warnings
warnings.filterwarnings("ignore")

import config as cfg

INTERPRETER = None
PBMODEL = None

def loadModel():

    global PBMODEL
    global INTERPRETER
    global INPUT_LAYER_INDEX
    global OUTPUT_LAYER_INDEX

    # Doe have to load te tflite or protobuf model?
    if cfg.MODEL_PATH.endswith('.tflite'):

        # Load TFLite model and allocate tensors.
        INTERPRETER = tflite.Interpreter(model_path=cfg.MODEL_PATH, num_threads=cfg.TFLITE_THREADS)
        INTERPRETER.allocate_tensors()

        # Get input and output tensors.
        input_details = INTERPRETER.get_input_details()
        output_details = INTERPRETER.get_output_details()

        # Get input tensor index
        INPUT_LAYER_INDEX = input_details[0]['index']
        OUTPUT_LAYER_INDEX = output_details[0]['index']

    else:

        # Load protobuf model
        # Note: This will throw a bunch of warnings about custom gradients
        # which we will ignore until TF lets us block them
        PBMODEL = keras.models.load_model(cfg.MODEL_PATH, compile=False)

def makeSample(sig):

    # Add batch axis
    sig = np.expand_dims(sig, 0)

    return [sig]

def flat_sigmoid(x, sensitivity=-1):
    return 1 / (1.0 + np.exp(sensitivity * np.clip(x, -15, 15)))

def predict(sample):

    global INTERPRETER

    # Does interpreter or keras model exist?
    if INTERPRETER == None and PBMODEL == None:
        loadModel()

    if PBMODEL == None:

        # Make a prediction (Audio only for now)
        INTERPRETER.set_tensor(INPUT_LAYER_INDEX, np.array(sample[0], dtype='float32'))
        INTERPRETER.invoke()
        prediction = INTERPRETER.get_tensor(OUTPUT_LAYER_INDEX)[0]

        return prediction

    else:

        # Make a prediction (Audio only for now)
        prediction = PBMODEL.predict(sample)[0]

        return prediction