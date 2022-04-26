import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np

import warnings
warnings.filterwarnings("ignore")

import config as cfg

# Import TFLite from runtrime or Tensorflow;
# import Keras if protobuf model; 
# NOTE: we have to use TFLite if we want to use 
# the metadata model or want to extract embeddings
try:
    import tflite_runtime.interpreter as tflite
except ModuleNotFoundError:
    from tensorflow import lite as tflite
if not cfg.MODEL_PATH.endswith('.tflite'):
   from tensorflow import keras

INTERPRETER = None
M_INTERPRETER = None
PBMODEL = None

def loadModel(class_output=True):

    global PBMODEL
    global INTERPRETER
    global INPUT_LAYER_INDEX
    global OUTPUT_LAYER_INDEX

    # Do we have to load the tflite or protobuf model?
    if cfg.MODEL_PATH.endswith('.tflite'):

        # Load TFLite model and allocate tensors.
        INTERPRETER = tflite.Interpreter(model_path=cfg.MODEL_PATH, num_threads=cfg.TFLITE_THREADS)
        INTERPRETER.allocate_tensors()

        # Get input and output tensors.
        input_details = INTERPRETER.get_input_details()
        output_details = INTERPRETER.get_output_details()

        # Get input tensor index
        INPUT_LAYER_INDEX = input_details[0]['index']

        # Get classification output or feature embeddings
        if class_output:
            OUTPUT_LAYER_INDEX = output_details[0]['index']
        else:
            OUTPUT_LAYER_INDEX = output_details[0]['index'] - 1

    else:

        # Load protobuf model
        # Note: This will throw a bunch of warnings about custom gradients
        # which we will ignore until TF lets us block them
        PBMODEL = keras.models.load_model(cfg.MODEL_PATH, compile=False)

def loadMetaModel():

    global M_INTERPRETER
    global M_INPUT_LAYER_INDEX
    global M_OUTPUT_LAYER_INDEX

    # Load TFLite model and allocate tensors.
    M_INTERPRETER = tflite.Interpreter(model_path=cfg.MDATA_MODEL_PATH, num_threads=cfg.TFLITE_THREADS)
    M_INTERPRETER.allocate_tensors()

    # Get input and output tensors.
    input_details = M_INTERPRETER.get_input_details()
    output_details = M_INTERPRETER.get_output_details()

    # Get input tensor index
    M_INPUT_LAYER_INDEX = input_details[0]['index']
    M_OUTPUT_LAYER_INDEX = output_details[0]['index']

def predictFilter(lat, lon, week):

    global M_INTERPRETER

    # Does interpreter exist?
    if M_INTERPRETER == None:
        loadMetaModel()

    # Prepare mdata as sample
    sample = np.expand_dims(np.array([lat, lon, week], dtype='float32'), 0)

    # Run inference
    M_INTERPRETER.set_tensor(M_INPUT_LAYER_INDEX, sample)
    M_INTERPRETER.invoke()

    return M_INTERPRETER.get_tensor(M_OUTPUT_LAYER_INDEX)[0]

def explore(lat, lon, week):

    # Make filter prediction
    l_filter = predictFilter(lat, lon, week)

    # Apply threshold
    l_filter = np.where(l_filter >= cfg.LOCATION_FILTER_THRESHOLD, l_filter, 0)

    # Zip with labels
    l_filter = list(zip(l_filter, cfg.LABELS))

    # Sort by filter value
    l_filter = sorted(l_filter, key=lambda x: x[0], reverse=True)

    return l_filter

def flat_sigmoid(x, sensitivity=-1):
    return 1 / (1.0 + np.exp(sensitivity * np.clip(x, -15, 15)))

def predict(sample):

    global INTERPRETER

    # Does interpreter or keras model exist?
    if INTERPRETER == None and PBMODEL == None:
        loadModel()

    if PBMODEL == None:

        # Reshape input tensor
        INTERPRETER.resize_tensor_input(INPUT_LAYER_INDEX, [len(sample), *sample[0].shape])
        INTERPRETER.allocate_tensors()

        # Make a prediction (Audio only for now)
        INTERPRETER.set_tensor(INPUT_LAYER_INDEX, np.array(sample, dtype='float32'))
        INTERPRETER.invoke()
        prediction = INTERPRETER.get_tensor(OUTPUT_LAYER_INDEX)

        return prediction

    else:

        # Make a prediction (Audio only for now)
        prediction = PBMODEL.predict(sample)

        return prediction

def embeddings(sample):

    global INTERPRETER

    # Does interpreter exist?
    if INTERPRETER == None:
        loadModel(False)

    # Reshape input tensor
    INTERPRETER.resize_tensor_input(INPUT_LAYER_INDEX, [len(sample), *sample[0].shape])
    INTERPRETER.allocate_tensors()

    # Extract feature embeddings
    INTERPRETER.set_tensor(INPUT_LAYER_INDEX, np.array(sample, dtype='float32'))
    INTERPRETER.invoke()
    features = INTERPRETER.get_tensor(OUTPUT_LAYER_INDEX)

    return features