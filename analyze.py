import os
import json
import operator
import argparse
import datetime

from multiprocessing import Pool

import numpy as np

import config as cfg
import audio
import model

def writeErrorLog(msg):

    with open(cfg.ERROR_LOG_FILE, 'a') as elog:
        elog.write(msg + '\n')

def parseInputFiles(path, allowed_filetypes=['wav', 'flac', 'mp3', 'ogg']):

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
        codes = json.load(cfile)

    return codes

def loadLabels():

    labels = []
    with open(cfg.LABELS_FILE, 'r') as lfile:
        for line in lfile.readlines():
            labels.append(line.replace('\n', ''))    

    return labels

def loadSpeciesList():

    slist = []
    if not cfg.SPECIES_LIST_FILE == None:
        with open(cfg.SPECIES_LIST_FILE, 'r') as sfile:
            for line in sfile.readlines():
                species = line.replace('\r', '').replace('\n', '')
                slist.append(species)

    return slist

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

def analyzeFile(entry):

    # Get file path and restore cfg
    fpath = entry[0]
    cfg.CODES = entry[1]
    cfg.LABELS = entry[2]
    cfg.SPECIES_LIST = entry[3]
    cfg.INPUT_PATH = entry[4]
    cfg.OUTPUT_PATH = entry[5]
    cfg.MIN_CONFIDENCE = entry[6]

    # Start time
    start_time = datetime.datetime.now()

    # Status
    print('Analyzing {}'.format(fpath), flush=True)

    # Open audio file and split into 3-second chunks
    chunks = getRawAudioFromFile(fpath)

    # If no chunks, show error and skip
    if len(chunks) == 0:
        msg = 'Error: Cannot open audio file {}'.format(fpath)
        print(msg, flush=True)
        writeErrorLog(msg)
        return

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
    if os.path.isdir(cfg.OUTPUT_PATH):
        fpath = fpath.replace(cfg.INPUT_PATH, '')
        fpath = fpath[1:] if fpath[0] in ['/', '\\'] else fpath
        saveAsSelectionTable(results, os.path.join(cfg.OUTPUT_PATH, fpath.rsplit('.', 1)[0] + '.BirdNET.selection.table.txt'))
    else:
        saveAsSelectionTable(results, cfg.OUTPUT_PATH)

    delta_time = (datetime.datetime.now() - start_time).total_seconds()
    print('Finished {} in {:.2f} seconds'.format(fpath, delta_time), flush=True)


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Analyze audio files with BirdNET')
    parser.add_argument('--i', default='example/', help='Path to input file or folder.')
    parser.add_argument('--o', default='example/', help='Path to output file or folder.')
    parser.add_argument('--lat', type=float, default=-1, help='Recording location latitude. Set -1 to ignore.')
    parser.add_argument('--lon', type=float, default=-1, help='Recording location longitude. Set -1 to ignore.')
    parser.add_argument('--week', type=int, default=-1, help='Week of the year when the recording was made. Values in [1, 48] (4 weeks per month). Set -1 to ignore for year-round species list.')
    parser.add_argument('--slist', default='example/', help='Path to species list file or folder. If folder is provided, species list needs to be named \"species_list.txt\". If lat and lon are provided, this list will be ignored.')
    parser.add_argument('--min_conf', type=float, default=0.1, help='Minimum confidence threshold. Values in [0.01, 0.99]. Defaults to 0.1.')
    parser.add_argument('--threads', type=int, default=4, help='Number of CPU threads.')

    args = parser.parse_args()

    # Load eBird codes, labels
    cfg.CODES = loadCodes()
    cfg.LABELS = loadLabels()

    # Load species list from location filter or provided list
    if args.lat == -1 and args.lon == -1:
        cfg.SPECIES_LIST_FILE = args.slist
        if os.path.isdir(args.slist):
            cfg.SPECIES_LIST_FILE = os.path.join(args.slist, 'species_list.txt')
        cfg.SPECIES_LIST = loadSpeciesList()
    else:
        l_filter = model.explore(args.lat, args.lon, args.week)
        cfg.SPECIES_LIST = []
        for s in l_filter:
            if s[0] >= cfg.LOCATION_FILTER_THRESHOLD:
                cfg.SPECIES_LIST.append(s[1])        
    print('Species list contains {} species'.format(len(cfg.SPECIES_LIST)))

    # Set input and output path
    # Comment out if you are not using args
    cfg.INPUT_PATH = args.i
    cfg.OUTPUT_PATH = args.o

    # Parse input files
    if os.path.isdir(cfg.INPUT_PATH):
        cfg.FILE_LIST = parseInputFiles(cfg.INPUT_PATH)  
    else:
        cfg.FILE_LIST = [cfg.INPUT_PATH]

    # Set number of threads
    if os.path.isdir(cfg.INPUT_PATH):
        cfg.CPU_THREADS = int(args.threads)
        cfg.TFLITE_THREADS = 1
    else:
        cfg.CPU_THREADS = 1
        cfg.TFLITE_THREADS = int(args.threads)

    # Set confidence threshold
    cfg.MIN_CONFIDENCE = max(0.01, min(0.99, float(args.min_conf)))

    # Add config items to each file list entry.
    # We have to do this for Windows which does not
    # support fork() and thus each process has to
    # have its own config. USE LINUX!
    flist = []
    for f in cfg.FILE_LIST:
        flist.append((f, cfg.CODES, cfg.LABELS, cfg.SPECIES_LIST, cfg.INPUT_PATH, cfg.OUTPUT_PATH, cfg.MIN_CONFIDENCE))

    # Analyze files   
    if cfg.CPU_THREADS < 2:
        for entry in flist:
            analyzeFile(entry)
    else:
        with Pool(cfg.CPU_THREADS) as p:
            p.map(analyzeFile, flist)


    # A few examples to test
    # python3 analyze.py --i example/ --o example/ --slist example/ --min_conf 0.5 --threads 4
    # python3 analyze.py --i example/soundscape.wav --o example/soundscape.BirdNET.selection.table.txt --slist example/species_list.txt --threads 8
    # python3 analyze.py --i example/ --o example/ --lat 50.8 --lon 12.9 --week 25
    