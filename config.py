#################
# Misc settings #
#################

# Random seed for gaussian noise
RANDOM_SEED = 42

##########################
# Model paths and config #
##########################

#MODEL_PATH = 'checkpoints/V2.2/BirdNET_GLOBAL_3K_V2.2_Model' # This will load the protobuf model
MODEL_PATH = 'checkpoints/V2.2/BirdNET_GLOBAL_3K_V2.2_Model_FP32.tflite'
MDATA_MODEL_PATH = 'checkpoints/V2.2/BirdNET_GLOBAL_3K_V2.2_MData_Model_FP16.tflite'
LABELS_FILE = 'checkpoints/V2.2/BirdNET_GLOBAL_3K_V2.2_Labels.txt'
TRANSLATED_LABELS_PATH = 'labels/V2.2'

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
# chunks shorter than 3 seconds will be padded with zeros
SIG_MINLEN = 1.0 

#####################
# Metadata settings #
#####################

LATITUDE = -1
LONGITUDE = -1
WEEK = -1
LOCATION_FILTER_THRESHOLD = 0.03

######################
# Inference settings #
######################

# If None or empty file, no custom species list will be used
# Note: Entries in this list have to match entries from the LABELS_FILE
# We use the 2021 eBird taxonomy for species names (Clements list)
CODES_FILE = 'eBird_taxonomy_codes_2021E.json'
SPECIES_LIST_FILE = 'example/species_list.txt' 

# File input path and output path for selection tables
INPUT_PATH = 'example/'
OUTPUT_PATH = 'example/'

# Number of threads to use for inference.
# Can be as high as number of CPUs in your system
CPU_THREADS = 8
TFLITE_THREADS = 1 

# False will output logits, True will convert to sigmoid activations
APPLY_SIGMOID = True 
SIGMOID_SENSITIVITY = 1.0

# Minimum confidence score to include in selection table 
# (be aware: if APPLY_SIGMOID = False, this no longer represents 
# probabilities and needs to be adjusted)
MIN_CONFIDENCE = 0.1 

# Number of samples to process at the same time. Higher values can increase
# processing speed, but will also increase memory usage.
# Might only be useful for GPU inference.
BATCH_SIZE = 1

# Specifies the output format. 'table' denotes a Raven selection table,
# 'audacity' denotes a TXT file with the same format as Audacity timeline labels
# 'csv' denotes a CSV file with start, end, species and confidence.
RESULT_TYPE = 'table'

#####################
# Misc runtime vars #
#####################
CODES = {}
LABELS = []
TRANSLATED_LABELS = []
SPECIES_LIST = []
ERROR_LOG_FILE = 'error_log.txt'

######################
# Get and set config #
######################

def getConfig():
    return {
        'RANDOM_SEED': RANDOM_SEED,
        'MODEL_PATH': MODEL_PATH,
        'MDATA_MODEL_PATH': MDATA_MODEL_PATH,
        'LABELS_FILE': LABELS_FILE,
        'SAMPLE_RATE': SAMPLE_RATE,
        'SIG_LENGTH': SIG_LENGTH,
        'SIG_OVERLAP': SIG_OVERLAP,
        'SIG_MINLEN': SIG_MINLEN,
        'LATITUDE': LATITUDE,
        'LONGITUDE': LONGITUDE,
        'WEEK': WEEK,
        'LOCATION_FILTER_THRESHOLD': LOCATION_FILTER_THRESHOLD,
        'CODES_FILE': CODES_FILE,
        'SPECIES_LIST_FILE': SPECIES_LIST_FILE,
        'INPUT_PATH': INPUT_PATH,
        'OUTPUT_PATH': OUTPUT_PATH,
        'CPU_THREADS': CPU_THREADS,
        'TFLITE_THREADS': TFLITE_THREADS,
        'APPLY_SIGMOID': APPLY_SIGMOID,
        'SIGMOID_SENSITIVITY': SIGMOID_SENSITIVITY,
        'MIN_CONFIDENCE': MIN_CONFIDENCE,
        'BATCH_SIZE': BATCH_SIZE,
        'RESULT_TYPE': RESULT_TYPE,
        'CODES': CODES,
        'LABELS': LABELS,
        'TRANSLATED_LABELS': TRANSLATED_LABELS,
        'SPECIES_LIST': SPECIES_LIST,
        'ERROR_LOG_FILE': ERROR_LOG_FILE
    }

def setConfig(c):

    global RANDOM_SEED
    global MODEL_PATH
    global MDATA_MODEL_PATH
    global LABELS_FILE
    global SAMPLE_RATE
    global SIG_LENGTH
    global SIG_OVERLAP
    global SIG_MINLEN
    global LATITUDE
    global LONGITUDE
    global WEEK
    global LOCATION_FILTER_THRESHOLD
    global CODES_FILE
    global SPECIES_LIST_FILE
    global INPUT_PATH
    global OUTPUT_PATH
    global CPU_THREADS
    global TFLITE_THREADS
    global APPLY_SIGMOID
    global SIGMOID_SENSITIVITY
    global MIN_CONFIDENCE
    global BATCH_SIZE
    global RESULT_TYPE
    global CODES
    global LABELS
    global TRANSLATED_LABELS
    global SPECIES_LIST
    global ERROR_LOG_FILE

    RANDOM_SEED = c['RANDOM_SEED']
    MODEL_PATH = c['MODEL_PATH']
    MDATA_MODEL_PATH = c['MDATA_MODEL_PATH']
    LABELS_FILE = c['LABELS_FILE']
    SAMPLE_RATE = c['SAMPLE_RATE']
    SIG_LENGTH = c['SIG_LENGTH']
    SIG_OVERLAP = c['SIG_OVERLAP']
    SIG_MINLEN = c['SIG_MINLEN']
    LATITUDE = c['LATITUDE']
    LONGITUDE = c['LONGITUDE']
    WEEK = c['WEEK']
    LOCATION_FILTER_THRESHOLD = c['LOCATION_FILTER_THRESHOLD']
    CODES_FILE = c['CODES_FILE']
    SPECIES_LIST_FILE = c['SPECIES_LIST_FILE']
    INPUT_PATH = c['INPUT_PATH']
    OUTPUT_PATH = c['OUTPUT_PATH']
    CPU_THREADS = c['CPU_THREADS']
    TFLITE_THREADS = c['TFLITE_THREADS']
    APPLY_SIGMOID = c['APPLY_SIGMOID']
    SIGMOID_SENSITIVITY = c['SIGMOID_SENSITIVITY']
    MIN_CONFIDENCE = c['MIN_CONFIDENCE']
    BATCH_SIZE = c['BATCH_SIZE']
    RESULT_TYPE = c['RESULT_TYPE']
    CODES = c['CODES']
    LABELS = c['LABELS']
    TRANSLATED_LABELS = c['TRANSLATED_LABELS']
    SPECIES_LIST = c['SPECIES_LIST']
    ERROR_LOG_FILE = c['ERROR_LOG_FILE']
