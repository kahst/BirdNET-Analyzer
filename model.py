import os
import warnings

import numpy as np

import config as cfg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

warnings.filterwarnings("ignore")

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

INTERPRETER: tflite.Interpreter = None
C_INTERPRETER: tflite.Interpreter = None
M_INTERPRETER: tflite.Interpreter = None
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

def loadCustomClassifier():

    global C_INTERPRETER
    global C_INPUT_LAYER_INDEX
    global C_OUTPUT_LAYER_INDEX

    # Load TFLite model and allocate tensors.
    C_INTERPRETER = tflite.Interpreter(model_path=cfg.CUSTOM_CLASSIFIER, num_threads=cfg.TFLITE_THREADS)
    C_INTERPRETER.allocate_tensors()

    # Get input and output tensors.
    input_details = C_INTERPRETER.get_input_details()
    output_details = C_INTERPRETER.get_output_details()

    # Get input tensor index
    C_INPUT_LAYER_INDEX = input_details[0]['index']

    # Get classification output
    C_OUTPUT_LAYER_INDEX = output_details[0]['index']

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

def buildLinearClassifier(num_labels, input_size, hidden_units=0):

    # import keras
    from tensorflow import keras

    # Build a simple one- or two-layer linear classifier
    model = keras.Sequential()

    # Input layer
    model.add(keras.layers.InputLayer(input_shape=(input_size,)))

    # Hidden layer
    if hidden_units > 0:
        model.add(keras.layers.Dense(hidden_units, activation='relu'))
        
    # Classification layer  
    model.add(keras.layers.Dense(num_labels))

    # Activation layer
    model.add(keras.layers.Activation('sigmoid'))

    return model

def trainLinearClassifier(classifier: keras.Sequential, x_train, y_train, epochs, batch_size, learning_rate, on_epoch_end=None) -> tuple[keras.Sequential, keras.callbacks.History]:

    # import keras
    from tensorflow import keras

    class FunctionCallback(keras.callbacks.Callback):
        def __init__(self, on_epoch_end=None) -> None:
            super().__init__()
            self.on_epoch_end_fn = on_epoch_end
        
        def on_epoch_end(self, epoch, logs=None):
            if self.on_epoch_end_fn:
                self.on_epoch_end_fn(epoch, logs)


    # Set random seed
    np.random.seed(cfg.RANDOM_SEED)

    # Shuffle data
    idx = np.arange(x_train.shape[0])
    np.random.shuffle(idx)
    x_train = x_train[idx]
    y_train = y_train[idx]
    
    # Random val split
    x_val = x_train[int(0.8 * x_train.shape[0]):]
    y_val = y_train[int(0.8 * y_train.shape[0]):]

    # Early stopping
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        FunctionCallback(on_epoch_end=on_epoch_end)
    ]
    
    # Cosine annealing lr schedule
    lr_schedule = keras.experimental.CosineDecay(learning_rate, epochs * x_train.shape[0] / batch_size)

    # Compile model
    classifier.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), 
                       loss='binary_crossentropy', 
                       metrics=keras.metrics.Precision(top_k=1, name='prec'))

    # Train model
    history = classifier.fit(x_train, 
                             y_train, 
                             epochs=epochs, 
                             batch_size=batch_size,
                             validation_data=(x_val, y_val), 
                             callbacks=callbacks)

    return classifier, history

def saveLinearClassifier(classifier, model_path, labels):

    # Make folders
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Remove activation layer
    classifier.pop()

    # Save model as tflite
    converter = tflite.TFLiteConverter.from_keras_model(classifier)
    tflite_model = converter.convert()
    open(model_path, "wb").write(tflite_model)

    # Save labels
    with open(model_path.replace('.tflite', '_Labels.txt'), 'w') as f:
        for label in labels:
            f.write(label + '\n')

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

    # Has custom classifier?
    if cfg.CUSTOM_CLASSIFIER != None:
        return predictWithCustomClassifier(sample)

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

def predictWithCustomClassifier(sample):

    global C_INTERPRETER

    # Does interpreter exist?
    if C_INTERPRETER == None:
        loadCustomClassifier()

    # Get embeddings
    feature_vector = embeddings(sample)

    # Reshape input tensor
    C_INTERPRETER.resize_tensor_input(C_INPUT_LAYER_INDEX, [len(feature_vector), *feature_vector[0].shape])
    C_INTERPRETER.allocate_tensors()

    # Make a prediction
    C_INTERPRETER.set_tensor(C_INPUT_LAYER_INDEX, np.array(feature_vector, dtype='float32'))
    C_INTERPRETER.invoke()
    prediction = C_INTERPRETER.get_tensor(C_OUTPUT_LAYER_INDEX)

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