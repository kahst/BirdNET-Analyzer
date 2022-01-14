import os
import json
import operator
import argparse

from multiprocessing import Pool

import numpy as np

import config as cfg
import audio
import model

def parseInputFiles(path, allowed_filetypes=['wav', 'flac', 'mp3']):

    # Get all files in directory with os.walk
    files = []
    for root, dirs, flist in os.walk(path):
        for f in flist:
            if f.rsplit('.', 1)[1] in allowed_filetypes:
                files.append(os.path.join(root, f))

    print('Found {} files to analyze'.format(len(files)))

    return sorted(files)

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

    # Make folder if it doesn't exist
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

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

    # Status
    print('Analyzing {}'.format(fpath), flush=True)

    # Open audio file and split into 3-second chunks
    chunks = getRawAudioFromFile(fpath)

    # Process each chunk
    start, end = 0, cfg.SIG_LENGTH
    results = {}
    for chunk in chunks:
        p = predict(chunk)

        # Assign scores to labels
        p_labels = dict(zip(cfg.LABELS, p))

        # Sort by score
        p_sorted =  sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)

        # Store top 5 results and advance indicies
        results[str(start) + '-' + str(end)] = p_sorted
        start += cfg.SIG_LENGTH - cfg.SIG_OVERLAP
        end = start + cfg.SIG_LENGTH

    # Save as selection table
    saveAsSelectionTable(results, os.path.join(cfg.OUTPUT_PATH, fpath.replace(cfg.INPUT_PATH, '')[1:].rsplit('.', 1)[0] + '.BirdNET.selection.table.txt'))

    # Status
    cfg.DONE_LIST.append(fpath)
    print('{}/{} files done.'.format(len(cfg.DONELIST), len(cfg.FILES)), flush=True)

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Analyze audio files with BirdNET')
    parser.add_argument('--i', default='example/', help='Path to input file folder.')
    parser.add_argument('--o', default='example/', help='Path for selection table output folder.')
    parser.add_argument('--slist', default='example/', help='Path to species list folder. Species list needs to be named \"species_list.txt\"')
    parser.add_argument('--threads', default=8, help='Number of CPU threads.')

    args = parser.parse_args()

    # Load eBird codes, labels
    loadCodes()
    loadLabels()

    # Load species list
    cfg.SPECIES_LIST_FILE = os.path.join(args.slist, 'species_list.txt') # remove this line if your not using args
    loadSpeciesList()
    print('Species list contains {} species'.format(len(cfg.SPECIES_LIST)))

    # Set input and output path
    # Comment out if you are not using args
    cfg.INPUT_PATH = args.i
    cfg.OUTPUT_PATH = args.o

    # Parse input files
    cfg.FILE_LIST = parseInputFiles(cfg.INPUT_PATH)  
    cfg.DONE_LIST = []  

    # Set number of threads
    cfg.CPU_THREADS = int(args.threads)

    # Analyze files
    pool = Pool(processes=cfg.CPU_THREADS)
    pool.map(analyzeFile, cfg.FILE_LIST)
    pool.close()
    