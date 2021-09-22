# Random seed for gaussian noise
RANDOM_SEED = 42

# Model paths and config
MODEL_PATH = 'checkpoints/BirdNET_HB_NFC_1K_Baseline_epoch_24.tflite'
LABELS_FILE = 'checkpoints/BirdNET_1K_V1_Labels.txt'

# Audio settings
SAMPLE_RATE = 48000 # We use a sample rate of 48kHz, so the model input size is (batch size, 48000 * 3) = (1, 144000)
SIG_LENGTH = 3.0 # We're using 3-second chunks
SIG_OVERLAP = 0 # Define overlap between consecutive chunks <3.0; 0 = no overlap
SIG_MINLEN = 3.0 # Define minimum length of audio chunk for prediction, chunks shorter than 3 seconds will be padded with noise

# Inference settings
SPECIES_LIST_FILE = 'example/species_list.txt' # If None or empty file, no custom species list will be used
TFLITE_THREADS = 4 # Can be as high as numbers of CPU in your system
APPLY_SIGMOID = True # False will output logits, True will convert to sigmoid activations
MIN_CONFIDENCE = 0.1 # Minimum confidence score to include in selection table