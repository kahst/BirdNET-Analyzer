"""Contains functions to use the BirdNET models."""

import os
import sys
import warnings

import numpy as np

import birdnet_analyzer.config as cfg
import birdnet_analyzer.utils as utils

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

warnings.filterwarnings("ignore")

# Import TFLite from runtime or Tensorflow;
# import Keras if protobuf model;
# NOTE: we have to use TFLite if we want to use
# the metadata model or want to extract embeddings
try:
    import tflite_runtime.interpreter as tflite  # type: ignore
except ModuleNotFoundError:
    from tensorflow import lite as tflite
if not cfg.MODEL_PATH.endswith(".tflite"):
    from tensorflow import keras

INTERPRETER: tflite.Interpreter = None
C_INTERPRETER: tflite.Interpreter = None
M_INTERPRETER: tflite.Interpreter = None
PBMODEL = None
C_PBMODEL = None


def resetCustomClassifier():
    global C_INTERPRETER
    global C_PBMODEL

    C_INTERPRETER = None
    C_PBMODEL = None


def loadModel(class_output=True):
    """Initializes the BirdNET Model.

    Args:
        class_output: Omits the last layer when False.
    """
    global PBMODEL
    global INTERPRETER
    global INPUT_LAYER_INDEX
    global OUTPUT_LAYER_INDEX

    # Do we have to load the tflite or protobuf model?
    if cfg.MODEL_PATH.endswith(".tflite"):
        # Load TFLite model and allocate tensors.
        INTERPRETER = tflite.Interpreter(
            model_path=os.path.join(SCRIPT_DIR, cfg.MODEL_PATH), num_threads=cfg.TFLITE_THREADS
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
        PBMODEL = keras.models.load_model(os.path.join(SCRIPT_DIR, cfg.MODEL_PATH), compile=False)


def loadCustomClassifier():
    """Loads the custom classifier."""
    global C_INTERPRETER
    global C_INPUT_LAYER_INDEX
    global C_OUTPUT_LAYER_INDEX
    global C_INPUT_SIZE
    global C_PBMODEL

    if cfg.CUSTOM_CLASSIFIER.endswith(".tflite"):
        # Load TFLite model and allocate tensors.
        C_INTERPRETER = tflite.Interpreter(model_path=cfg.CUSTOM_CLASSIFIER, num_threads=cfg.TFLITE_THREADS)
        C_INTERPRETER.allocate_tensors()

        # Get input and output tensors.
        input_details = C_INTERPRETER.get_input_details()
        output_details = C_INTERPRETER.get_output_details()

        # Get input tensor index
        C_INPUT_LAYER_INDEX = input_details[0]["index"]

        C_INPUT_SIZE = input_details[0]["shape"][-1]

        # Get classification output
        C_OUTPUT_LAYER_INDEX = output_details[0]["index"]
    else:
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")

        C_PBMODEL = tf.saved_model.load(cfg.CUSTOM_CLASSIFIER)


def loadMetaModel():
    """Loads the model for species prediction.

    Initializes the model used to predict species list, based on coordinates and week of year.
    """
    global M_INTERPRETER
    global M_INPUT_LAYER_INDEX
    global M_OUTPUT_LAYER_INDEX

    # Load TFLite model and allocate tensors.
    M_INTERPRETER = tflite.Interpreter(
        model_path=os.path.join(SCRIPT_DIR, cfg.MDATA_MODEL_PATH), num_threads=cfg.TFLITE_THREADS
    )
    M_INTERPRETER.allocate_tensors()

    # Get input and output tensors.
    input_details = M_INTERPRETER.get_input_details()
    output_details = M_INTERPRETER.get_output_details()

    # Get input tensor index
    M_INPUT_LAYER_INDEX = input_details[0]["index"]
    M_OUTPUT_LAYER_INDEX = output_details[0]["index"]


def buildLinearClassifier(num_labels, input_size, hidden_units=0, dropout=0.0):
    """Builds a classifier.

    Args:
        num_labels: Output size.
        input_size: Size of the input.
        hidden_units: If > 0, creates another hidden layer with the given number of units.

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
        # Dropout layer?
        if dropout > 0:
            model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(hidden_units, activation="relu"))

    # Dropout layer?
    if dropout > 0:
        model.add(keras.layers.Dropout(dropout))

    # Classification layer
    model.add(keras.layers.Dense(num_labels))

    # Activation layer
    model.add(keras.layers.Activation("sigmoid"))

    return model


def trainLinearClassifier(
    classifier,
    x_train,
    y_train,
    epochs,
    batch_size,
    learning_rate,
    val_split,
    upsampling_ratio,
    upsampling_mode,
    train_with_mixup,
    train_with_label_smoothing,
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
    np.random.seed(cfg.RANDOM_SEED)

    # Shuffle data
    idx = np.arange(x_train.shape[0])
    np.random.shuffle(idx)
    x_train = x_train[idx]
    y_train = y_train[idx]

    # Random val split
    if not cfg.MULTI_LABEL:
        x_train, y_train, x_val, y_val = utils.random_split(x_train, y_train, val_split)
    else:
        x_train, y_train, x_val, y_val = utils.random_multilabel_split(x_train, y_train, val_split)

    print(
        f"Training on {x_train.shape[0]} samples, validating on {x_val.shape[0]} samples.",
        flush=True,
    )

    # Upsample training data
    if upsampling_ratio > 0:
        x_train, y_train = utils.upsampling(x_train, y_train, upsampling_ratio, upsampling_mode)
        print(f"Upsampled training data to {x_train.shape[0]} samples.", flush=True)

    # Apply mixup to training data
    if train_with_mixup and not cfg.BINARY_CLASSIFICATION:
        x_train, y_train = utils.mixup(x_train, y_train)

    # Apply label smoothing
    if train_with_label_smoothing and not cfg.BINARY_CLASSIFICATION:
        y_train = utils.label_smoothing(y_train)

    # Early stopping
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            verbose=1,
            start_from_epoch=5,
            restore_best_weights=True,
        ),
        FunctionCallback(on_epoch_end=on_epoch_end),
    ]

    # Cosine annealing lr schedule
    lr_schedule = keras.experimental.CosineDecay(learning_rate, epochs * x_train.shape[0] / batch_size)

    optimizer_cls = keras.optimizers.legacy.Adam if sys.platform == "darwin" else keras.optimizers.Adam

    # Compile model
    classifier.compile(
        optimizer=optimizer_cls(learning_rate=lr_schedule),
        loss=custom_loss,
        metrics=[
            keras.metrics.AUC(
                curve="PR",
                multi_label=cfg.MULTI_LABEL,
                name="AUPRC",
                num_labels=y_train.shape[1] if cfg.MULTI_LABEL else None,
                from_logits=True,
            ),
            keras.metrics.AUC(
                curve="ROC",
                multi_label=cfg.MULTI_LABEL,
                name="AUROC",
                num_labels=y_train.shape[1] if cfg.MULTI_LABEL else None,
                from_logits=True,
            ),
        ],
    )

    # Train model
    history = classifier.fit(
        x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=callbacks
    )

    return classifier, history


def saveLinearClassifier(classifier, model_path: str, labels: list[str], mode="replace"):
    """Saves a custom classifier on the hard drive.

    Saves the classifier as a tflite model, as well as the used labels in a .txt.

    Args:
        classifier: The custom classifier.
        model_path: Path the model will be saved at.
        labels: List of labels used for the classifier.
    """
    import tensorflow as tf

    global PBMODEL

    tf.get_logger().setLevel("ERROR")

    if PBMODEL is None:
        PBMODEL = tf.keras.models.load_model(os.path.join(SCRIPT_DIR, cfg.PB_MODEL), compile=False)

    saved_model = PBMODEL

    # Remove activation layer
    classifier.pop()

    if mode == "replace":
        combined_model = tf.keras.Sequential([saved_model.embeddings_model, classifier], "basic")
    elif mode == "append":
        intermediate = classifier(saved_model.model.get_layer("GLOBAL_AVG_POOL").output)

        output = tf.keras.layers.concatenate([saved_model.model.output, intermediate], name="combined_output")

        combined_model = tf.keras.Model(inputs=saved_model.model.input, outputs=output)
    else:
        raise ValueError("Model save mode must be either 'replace' or 'append'")

    # Append .tflite if necessary
    if not model_path.endswith(".tflite"):
        model_path += ".tflite"

    # Make folders
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save model as tflite
    converter = tf.lite.TFLiteConverter.from_keras_model(combined_model)
    tflite_model = converter.convert()
    open(model_path, "wb").write(tflite_model)

    if mode == "append":
        labels = [*utils.readLines(os.path.join(SCRIPT_DIR, cfg.LABELS_FILE)), *labels]

    # Save labels
    with open(model_path.replace(".tflite", "_Labels.txt"), "w", encoding="utf-8") as f:
        for label in labels:
            f.write(label + "\n")

    utils.save_model_params(model_path.replace(".tflite", "_Params.csv"))


def save_raven_model(classifier, model_path, labels: list[str], mode="replace"):
    import csv
    import json

    import tensorflow as tf

    global PBMODEL

    tf.get_logger().setLevel("ERROR")

    if PBMODEL is None:
        PBMODEL = tf.keras.models.load_model(os.path.join(SCRIPT_DIR, cfg.PB_MODEL), compile=False)

    saved_model = PBMODEL

    if mode == "replace":
        combined_model = tf.keras.Sequential([saved_model.embeddings_model, classifier], "basic")
    elif mode == "append":
        # Remove activation layer
        classifier.pop()
        intermediate = classifier(saved_model.model.get_layer("GLOBAL_AVG_POOL").output)

        output = tf.keras.layers.concatenate([saved_model.model.output, intermediate], name="combined_output")

        combined_model = tf.keras.Model(inputs=saved_model.model.input, outputs=output)
    else:
        raise ValueError("Model save mode must be either 'replace' or 'append'")

    # Make signatures
    class SignatureModule(tf.Module):
        def __init__(self, keras_model):
            super().__init__()
            self.model = keras_model

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, 144000], dtype=tf.float32)])
        def basic(self, inputs):
            return {"scores": self.model(inputs)}

    smodel = SignatureModule(combined_model)
    signatures = {
        "basic": smodel.basic,
    }

    # Save signature model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model_path = model_path[:-7] if model_path.endswith(".tflite") else model_path
    tf.saved_model.save(smodel, model_path, signatures=signatures)

    if mode == "append":
        labels = [*utils.readLines(os.path.join(SCRIPT_DIR, cfg.LABELS_FILE)), *labels]

    # Save label file
    labelIds = [label[:4].replace(" ", "") + str(i) for i, label in enumerate(labels, 1)]
    labels_dir = os.path.join(model_path, "labels")

    os.makedirs(labels_dir, exist_ok=True)

    with open(os.path.join(labels_dir, "label_names.csv"), "w", newline="") as labelsfile:
        labelwriter = csv.writer(labelsfile)
        labelwriter.writerows(zip(labelIds, labels))

    # Save class names file
    classes_dir = os.path.join(model_path, "classes")

    os.makedirs(classes_dir, exist_ok=True)

    with open(os.path.join(classes_dir, "classes.csv"), "w", newline="") as classesfile:
        classeswriter = csv.writer(classesfile)
        for labelId in labelIds:
            classeswriter.writerow((labelId, 0.25, cfg.SIG_FMIN, cfg.SIG_FMAX, False))

    # Save model config
    model_config = os.path.join(model_path, "model_config.json")

    with open(model_config, "w") as modelconfigfile:
        modelconfig = {
            "specVersion": 1,
            "modelDescription": "Custom classifier trained with BirdNET "
            + cfg.MODEL_VERSION
            + " embeddings.\n"
            + "BirdNET was developed by the K. Lisa Yang Center for Conservation Bioacoustics"
            + "at the Cornell Lab of Ornithology in collaboration with Chemnitz University of Technology.\n\n"
            + "https://birdnet.cornell.edu",
            "modelTypeConfig": {"modelType": "RECOGNITION"},
            "signatures": [
                {
                    "signatureName": "basic",
                    "modelInputs": [
                        {
                            "inputName": "inputs",
                            "sampleRate": 48000.0,
                            "inputConfig": ["batch", "samples"],
                        }
                    ],
                    "modelOutputs": [{"outputName": "scores", "outputType": "SCORES"}],
                }
            ],
            "globalSemanticKeys": labelIds,
        }
        json.dump(modelconfig, modelconfigfile, indent=2)

        model_params = os.path.join(model_path, "model_params.csv")

        utils.save_model_params(model_params)


def predictFilter(lat, lon, week):
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
    if M_INTERPRETER is None:
        loadMetaModel()

    # Prepare mdata as sample
    sample = np.expand_dims(np.array([lat, lon, week], dtype="float32"), 0)

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
    l_filter = predictFilter(lat, lon, week)

    # Apply threshold
    l_filter = np.where(l_filter >= cfg.LOCATION_FILTER_THRESHOLD, l_filter, 0)

    # Zip with labels
    l_filter = list(zip(l_filter, cfg.LABELS))

    # Sort by filter value
    l_filter = sorted(l_filter, key=lambda x: x[0], reverse=True)

    return l_filter


def custom_loss(y_true, y_pred, epsilon=1e-7):
    """Custom loss function that also estimated loss for negative labels.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        epsilon: Epsilon value to avoid log(0).

    Returns:
        The loss.
    """

    import tensorflow.keras.backend as K

    # Calculate loss for positive labels with epsilon
    positive_loss = -K.sum(y_true * K.log(K.clip(y_pred, epsilon, 1.0 - epsilon)), axis=-1)

    # Calculate loss for negative labels with epsilon
    negative_loss = -K.sum((1 - y_true) * K.log(K.clip(1 - y_pred, epsilon, 1.0 - epsilon)), axis=-1)

    # Combine both loss terms
    total_loss = positive_loss + negative_loss

    return total_loss


def flat_sigmoid(x, sensitivity=-1):
    return 1 / (1.0 + np.exp(sensitivity * np.clip(x, -15, 15)))


def predict(sample):
    """Uses the main net to predict a sample.

    Args:
        sample: Audio sample.

    Returns:
        The prediction scores for the sample.
    """
    # Has custom classifier?
    if cfg.CUSTOM_CLASSIFIER is not None:
        return predictWithCustomClassifier(sample)

    global INTERPRETER

    # Does interpreter or keras model exist?
    if INTERPRETER is None and PBMODEL is None:
        loadModel()

    if PBMODEL is None:
        # Reshape input tensor
        INTERPRETER.resize_tensor_input(INPUT_LAYER_INDEX, [len(sample), *sample[0].shape])
        INTERPRETER.allocate_tensors()

        # Make a prediction (Audio only for now)
        INTERPRETER.set_tensor(INPUT_LAYER_INDEX, np.array(sample, dtype="float32"))
        INTERPRETER.invoke()
        prediction = INTERPRETER.get_tensor(OUTPUT_LAYER_INDEX)

        return prediction

    else:
        # Make a prediction (Audio only for now)
        prediction = PBMODEL.basic(sample)["scores"]

        return prediction


def predictWithCustomClassifier(sample):
    """Uses the custom classifier to make a prediction.

    Args:
        sample: Audio sample.

    Returns:
        The prediction scores for the sample.
    """
    global C_INTERPRETER
    global C_INPUT_SIZE
    global C_PBMODEL

    # Does interpreter exist?
    if C_INTERPRETER is None and C_PBMODEL is None:
        loadCustomClassifier()

    if C_PBMODEL is None:
        vector = embeddings(sample) if C_INPUT_SIZE != 144000 else sample

        # Reshape input tensor
        C_INTERPRETER.resize_tensor_input(C_INPUT_LAYER_INDEX, [len(vector), *vector[0].shape])
        C_INTERPRETER.allocate_tensors()

        # Make a prediction
        C_INTERPRETER.set_tensor(C_INPUT_LAYER_INDEX, np.array(vector, dtype="float32"))
        C_INTERPRETER.invoke()
        prediction = C_INTERPRETER.get_tensor(C_OUTPUT_LAYER_INDEX)

        return prediction
    else:
        prediction = C_PBMODEL.basic(sample)["scores"]

        return prediction


def embeddings(sample):
    """Extracts the embeddings for a sample.

    Args:
        sample: Audio samples.

    Returns:
        The embeddings.
    """
    global INTERPRETER

    # Does interpreter exist?
    if INTERPRETER is None:
        loadModel(False)

    # Reshape input tensor
    INTERPRETER.resize_tensor_input(INPUT_LAYER_INDEX, [len(sample), *sample[0].shape])
    INTERPRETER.allocate_tensors()

    # Extract feature embeddings
    INTERPRETER.set_tensor(INPUT_LAYER_INDEX, np.array(sample, dtype="float32"))
    INTERPRETER.invoke()
    features = INTERPRETER.get_tensor(OUTPUT_LAYER_INDEX)

    return features
