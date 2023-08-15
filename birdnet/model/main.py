"""Contains functions to use the BirdNET models.
"""
import os
import warnings

import numpy

from birdnet._paths import ROOT_PATH
from birdnet.configuration import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

warnings.filterwarnings("ignore")

# Import TFLite from runtime or Tensorflow;
# import Keras if protobuf model;
# NOTE: we have to use TFLite if we want to use
# the metadata model or want to extract embeddings
try:
    import tflite_runtime.interpreter as tflite
except ModuleNotFoundError:
    from tensorflow import lite as tflite
if not config.MODEL_PATH.endswith(".tflite"):
    from tensorflow import keras

INTERPRETER: tflite.Interpreter = None
C_INTERPRETER: tflite.Interpreter = None
M_INTERPRETER: tflite.Interpreter = None
PBMODEL = None


def load_model(class_output=True):
    """Initializes the BirdNET Model.

    Args:
        class_output: Omits the last layer when False.
    """
    global PBMODEL
    global INTERPRETER
    global INPUT_LAYER_INDEX
    global OUTPUT_LAYER_INDEX

    # Do we have to load the tflite or protobuf model?
    if config.MODEL_PATH.endswith(".tflite"):
        # Load TFLite model and allocate tensors.
        INTERPRETER = tflite.Interpreter(
            model_path=str(ROOT_PATH / config.MODEL_PATH),
            num_threads=config.TFLITE_THREADS,
        )
        INTERPRETER.allocate_tensors()

        # Get input and output tensors.
        input_details = INTERPRETER.get_input_details()
        output_details = INTERPRETER.get_output_details()

        # Get input tensor index
        INPUT_LAYER_INDEX = input_details[0]["index"]

        # Get classification output or feature embeddings
        if class_output:
            OUTPUT_LAYER_INDEX = output_details[0]["index"]
        else:
            OUTPUT_LAYER_INDEX = output_details[0]["index"] - 1

    else:
        # Load protobuf model
        # Note: This will throw a bunch of warnings about custom gradients
        # which we will ignore until TF lets us block them
        PBMODEL = keras.models.load_model(config.MODEL_PATH, compile=False)


def load_custom_classifier():
    """Loads the custom classifier."""
    global C_INTERPRETER
    global C_INPUT_LAYER_INDEX
    global C_OUTPUT_LAYER_INDEX

    # Load TFLite model and allocate tensors.
    C_INTERPRETER = tflite.Interpreter(
        model_path=config.CUSTOM_CLASSIFIER,
        num_threads=config.TFLITE_THREADS,
    )
    C_INTERPRETER.allocate_tensors()

    # Get input and output tensors.
    input_details = C_INTERPRETER.get_input_details()
    output_details = C_INTERPRETER.get_output_details()

    # Get input tensor index
    C_INPUT_LAYER_INDEX = input_details[0]["index"]

    # Get classification output
    C_OUTPUT_LAYER_INDEX = output_details[0]["index"]


def load_meta_model():
    """Loads the model for species prediction.

    Initializes the model used to predict species list, based on coordinates
    and week of year.
    """
    global M_INTERPRETER
    global M_INPUT_LAYER_INDEX
    global M_OUTPUT_LAYER_INDEX

    # Load TFLite model and allocate tensors.
    M_INTERPRETER = tflite.Interpreter(
        model_path=ROOT_PATH / config.MDATA_MODEL_PATH,
        num_threads=config.TFLITE_THREADS,
    )
    M_INTERPRETER.allocate_tensors()

    # Get input and output tensors.
    input_details = M_INTERPRETER.get_input_details()
    output_details = M_INTERPRETER.get_output_details()

    # Get input tensor index
    M_INPUT_LAYER_INDEX = input_details[0]["index"]
    M_OUTPUT_LAYER_INDEX = output_details[0]["index"]


def build_linear_classifier(num_labels, input_size, hidden_units=0):
    """Builds a classifier.

    Args:
        num_labels: Output size.
        input_size: Size of the input.
        hidden_units: If > 0, creates another hidden layer with the given
        number of units.

    Returns:
        A new classifier.
    """
    # import keras
    from tensorflow import keras

    # Build a simple one- or two-layer linear classifier
    model = keras.Sequential()

    # Input layer
    model.add(keras.layers.InputLayer(input_shape=(input_size,)))

    # Hidden layer
    if hidden_units > 0:
        model.add(keras.layers.Dense(hidden_units, activation="relu"))

    # Classification layer
    model.add(keras.layers.Dense(num_labels))

    # Activation layer
    model.add(keras.layers.Activation("sigmoid"))

    return model


def train_linear_classifier(
    classifier,
    x_train,
    y_train,
    epochs,
    batch_size,
    learning_rate,
    on_epoch_end=None,
):
    """Trains a custom classifier.

    Trains a new classifier for BirdNET based on the given data.

    Args:
        classifier: The classifier to be trained.
        x_train: Samples.
        y_train: Labels.
        epochs: Number of epochs to train.
        batch_size: Batch size.
        learning_rate: The learning rate during training.
        on_epoch_end: Optional callback `function(epoch, logs)`.

    Returns:
        (classifier, history)
    """
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
    numpy.random.seed(config.RANDOM_SEED)

    # Shuffle data
    idx = numpy.arange(x_train.shape[0])
    numpy.random.shuffle(idx)
    x_train = x_train[idx]
    y_train = y_train[idx]

    # Random val split
    x_val = x_train[int(0.8 * x_train.shape[0]) :]
    y_val = y_train[int(0.8 * y_train.shape[0]) :]

    # Early stopping
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        FunctionCallback(on_epoch_end=on_epoch_end),
    ]

    # Cosine annealing lr schedule
    lr_schedule = keras.experimental.CosineDecay(
        learning_rate,
        epochs * x_train.shape[0] / batch_size,
    )

    # Compile model
    classifier.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="binary_crossentropy",
        metrics=keras.metrics.Precision(top_k=1, name="prec"),
    )

    # Train model
    history = classifier.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
    )

    return classifier, history


def save_linear_classifier(classifier, model_path, labels):
    """Saves a custom classifier on the hard drive.

    Saves the classifier as a tflite model, as well as the used labels in a
    .txt.

    Args:
        classifier: The custom classifier.
        model_path: Path the model will be saved at.
        labels: List of labels used for the classifier.
    """
    # Make folders
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Remove activation layer
    classifier.pop()

    # Save model as tflite
    converter = tflite.TFLiteConverter.from_keras_model(classifier)
    tflite_model = converter.convert()
    open(model_path, "wb").write(tflite_model)

    # Save labels
    with open(model_path.replace(".tflite", "_Labels.txt"), "w") as f:
        for label in labels:
            f.write(label + "\n")


def predict_filter(lat, lon, week):
    """Predicts the probability for each species.

    Args:
        lat: The latitude.
        lon: The longitude.
        week: The week of the year [1-48]. Use -1 for yearlong.

    Returns:
        A list of probabilities for all species.
    """
    global M_INTERPRETER

    # Does interpreter exist?
    if M_INTERPRETER == None:
        load_meta_model()

    # Prepare mdata as sample
    sample = \
        numpy.expand_dims(numpy.array([lat, lon, week], dtype="float32"), 0)

    # Run inference
    M_INTERPRETER.set_tensor(M_INPUT_LAYER_INDEX, sample)
    M_INTERPRETER.invoke()

    return M_INTERPRETER.get_tensor(M_OUTPUT_LAYER_INDEX)[0]


def explore(lat: float, lon: float, week: int):
    """Predicts the species list.

    Predicts the species list based on the coordinates and week of year.

    Args:
        lat: The latitude.
        lon: The longitude.
        week: The week of the year [1-48]. Use -1 for yearlong.

    Returns:
        A sorted list of tuples with the score and the species.
    """
    # Make filter prediction
    l_filter = predict_filter(lat, lon, week)

    # Apply threshold
    l_filter = \
        numpy.where(l_filter >= config.LOCATION_FILTER_THRESHOLD, l_filter, 0)

    # Zip with labels
    l_filter = list(zip(l_filter, config.LABELS))

    # Sort by filter value
    l_filter = sorted(l_filter, key=lambda x: x[0], reverse=True)

    return l_filter


def flat_sigmoid(x, sensitivity=-1):
    return 1 / (1.0 + numpy.exp(sensitivity * numpy.clip(x, -15, 15)))


def predict_by_sample(sample):
    """Uses the main net to predict a sample.

    Args:
        sample: Audio sample.

    Returns:
        The prediction scores for the sample.
    """
    # Has custom classifier?
    if config.CUSTOM_CLASSIFIER != None:
        return predict_with_custom_classifier(sample)

    global INTERPRETER

    # Does interpreter or keras model exist?
    if INTERPRETER == None and PBMODEL == None:
        load_model()

    if PBMODEL == None:
        # Reshape input tensor
        INTERPRETER.resize_tensor_input(
            INPUT_LAYER_INDEX,
            [len(sample), *sample[0].shape],
        )
        INTERPRETER.allocate_tensors()

        # Make a prediction (Audio only for now)
        INTERPRETER.set_tensor(
            INPUT_LAYER_INDEX,
            numpy.array(sample, dtype="float32"),
        )
        INTERPRETER.invoke()
        prediction = INTERPRETER.get_tensor(OUTPUT_LAYER_INDEX)

        return prediction

    else:
        # Make a prediction (Audio only for now)
        prediction = PBMODEL.predict(sample)

        return prediction


def predict_with_custom_classifier(sample):
    """Uses the custom classifier to make a prediction.

    Args:
        sample: Audio sample.

    Returns:
        The prediction scores for the sample.
    """
    global C_INTERPRETER

    # Does interpreter exist?
    if C_INTERPRETER == None:
        load_custom_classifier()

    # Get embeddings
    feature_vector = extract_embeddings(sample)

    # Reshape input tensor
    C_INTERPRETER.resize_tensor_input(
        C_INPUT_LAYER_INDEX,
        [len(feature_vector), *feature_vector[0].shape],
    )
    C_INTERPRETER.allocate_tensors()

    # Make a prediction
    C_INTERPRETER.set_tensor(
        C_INPUT_LAYER_INDEX,
        numpy.array(feature_vector, dtype="float32"),
    )
    C_INTERPRETER.invoke()
    prediction = C_INTERPRETER.get_tensor(C_OUTPUT_LAYER_INDEX)

    return prediction


def extract_embeddings(sample):
    """Extracts the embeddings for a sample.

    Args:
        sample: Audio samples.

    Returns:
        The embeddings.
    """
    global INTERPRETER

    # Does interpreter exist?
    if INTERPRETER == None:
        load_model(False)

    # Reshape input tensor
    INTERPRETER.resize_tensor_input(
        INPUT_LAYER_INDEX,
        [len(sample), *sample[0].shape],
    )
    INTERPRETER.allocate_tensors()

    # Extract feature embeddings
    INTERPRETER.set_tensor(
        INPUT_LAYER_INDEX,
        numpy.array(sample, dtype="float32"),
    )
    INTERPRETER.invoke()
    features = INTERPRETER.get_tensor(OUTPUT_LAYER_INDEX)

    return features
