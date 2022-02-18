import json
import operator

import numpy as np
from pathlib import Path
import multiprocessing
from joblib import Parallel, delayed
#from tqdm import tqdm

import config as cfg
import audio
import model



def loadCodes():

    with open('eBird_taxonomy_codes_2021E.json', 'r') as cfile:
        cfg.CODES = json.load(cfile)

def loadLabels():

    cfg.LABELS = []
    with open(cfg.LABELS_FILE, 'r') as lfile:
        for line in lfile.readlines():
            cfg.LABELS.append(line.replace('\n', ''))    

def loadSpeciesList():

    cfg.SPECIES_LIST = []
    if not cfg.SPECIES_LIST_FILE == None:
        with open(cfg.SPECIES_LIST_FILE, 'r') as sfile:
            for line in sfile.readlines():
                species = line.replace('\r', '').replace('\n', '')
                cfg.SPECIES_LIST.append(species)

def saveAsSelectionTable(r, path):

    # Selection table
    stable = ''

    # Raven selection header
    header = 'Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tSpecies Code\tCommon Name\tConfidence\n'
    selection_id = 0

    # Write header
    stable += header
    
    # Extract valid predictions for every timestamp
    for timestamp in sorted(r):
        rstring = ''
        start, end = timestamp.split('-')
        for c in r[timestamp]:
            if c[1] > cfg.MIN_CONFIDENCE and c[0] in cfg.CODES and (c[0] in cfg.SPECIES_LIST or len(cfg.SPECIES_LIST) == 0):
                selection_id += 1
                rstring += str(selection_id) + '\tSpectrogram 1\t1\t'
                rstring += str(start) + '\t' + str(end) + '\t' + str(150) + '\t' + str(15000) + '\t'
                rstring += cfg.CODES[c[0]] + '\t' + c[0].split('_')[1] + '\t' + str(c[1]) + '\n'

        # Write result string to file
        if len(rstring) > 0:
            stable += rstring

    # Save as file
    with open(path, 'w') as stfile:
        stfile.write(stable)

def getRawAudioFromFile(fpath):

    # Open file
    sig, rate = audio.openAudioFile(fpath, cfg.SAMPLE_RATE)

    # Split into raw audio chunks
    chunks = audio.splitSignal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)
    print("number of chunks", len(chunks))

    return chunks

def predict(sig):

    # Prepare sample and pass through model
    data = model.makeSample(sig)
    prediction = model.predict(data)

    # Logits or sigmoid activations?
    if cfg.APPLY_SIGMOID:
        prediction = model.flat_sigmoid(np.array(prediction))

    return prediction

def analyzeFile(fpath):

    # Open audio file and split into 3-second chunks
    chunks = getRawAudioFromFile(fpath)

    # Process each chunk
    start, end = 0, cfg.SIG_LENGTH - cfg.SIG_OVERLAP
    results = {}
    for chunk in chunks:
        p = predict(chunk)

        # Assign scores to labels
        p_labels = dict(zip(cfg.LABELS, p))

        # Sort by score
        p_sorted =  sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)

        # Store top 5 results and advance indicies
        results[str(start) + '-' + str(end)] = p_sorted
        start = end
        end += cfg.SIG_LENGTH - cfg.SIG_OVERLAP

    return results

def analyze_files(path_src, dest_dir):
    loadCodes()
    loadLabels()
    loadSpeciesList()
    print(dest_dir)
    #print(path_src.parts[-2])
    print(path_src.stem)
    #dsn =  Path(dest_dir, path_src.parts[-2], path_src.stem).with_suffix(".txt")   
    dsn =  Path(dest_dir, path_src.stem).with_suffix(".txt")   
    print(dsn) 
   # Analyze a file
    result = analyzeFile(path_src)

    # Generate Raven selection table
    saveAsSelectionTable(result, dsn)
"""
if __name__ == '__main__':

    # Load eBird codes, labels and custom species list
    loadCodes()
    loadLabels()
    loadSpeciesList()

    # Analyze a file
    result = analyzeFile('/media/oekofor/Daten_1/002_WiWa2/Daten/Aufnahmen/SN001_audio/SN001_2019_02_21_17_53_07_evening.wav')

    # Generate Raven selection table
    saveAsSelectionTable(result, 'example/soundscape_wiwa.txt')

"""
