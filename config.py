#################
# Misc settings #
#################

# Random seed for gaussian noise
RANDOM_SEED = 42

##########################
# Model paths and config #
##########################

MODEL_PATH = 'checkpoints/BirdNET_1K_V1.1_Model_FP32.tflite'
LABELS_FILE = 'checkpoints/BirdNET_1K_V1.1_Labels.txt'

##################
# Audio settings #
##################

# We use a sample rate of 48kHz, so the model input size is 
# (batch size, 48000 kHz * 3 seconds) = (1, 144000)
# Recordings will be resampled automatically.
SAMPLE_RATE = 48000 

# We're using 3-second chunks
SIG_LENGTH = 3.0 

# Define overlap between consecutive chunks <3.0; 0 = no overlap
SIG_OVERLAP = 0 

# Define minimum length of audio chunk for prediction, 
# chunks shorter than 3 seconds will be padded with noise
SIG_MINLEN = 3.0 

######################
# Inference settings #
######################

# If None or empty file, no custom species list will be used
# Note: Entries in this list have to match entries from the LABELS_FILE
# We use the 2021 eBird taxonomy for species names (Clements list)
SPECIES_LIST_FILE = 'example/species_list.txt' 

# Number of threads to use for inference.
# Can be as high as number of CPUs in your system
TFLITE_THREADS = 4 

# False will output logits, True will convert to sigmoid activations
APPLY_SIGMOID = True 

# Minimum confidence score to include in selection table 
# (be aware: if APPLY_SIGMOID = False, this no longer represents 
# probabilities and needs to be adjusted)
MIN_CONFIDENCE = 0.1 