import sys
import os
import argparse
import datetime
import numpy as np
import traceback

from multiprocessing import Pool

import config as cfg
import analyze
import model

def clearErrorLog():

    if os.path.isfile(cfg.ERROR_LOG_FILE):
        os.remove(cfg.ERROR_LOG_FILE)

def writeErrorLog(msg):

    with open(cfg.ERROR_LOG_FILE, 'a') as elog:
        elog.write(msg + '\n')

def saveAsEmbeddingsFile(results, fpath):

    # Write embeddings to file
    with open(fpath, 'w') as f:
        for timestamp in results:
            f.write(timestamp.replace('-', '\t') + '\t' + ','.join(map(str, results[timestamp])) + '\n')

def analyzeFile(item):

    # Get file path and restore cfg
    fpath = item[0]
    cfg.setConfig(item[1])

    # Start time
    start_time = datetime.datetime.now()

    # Status
    print('Analyzing {}'.format(fpath), flush=True)

    # Open audio file and split into 3-second chunks
    chunks = analyze.getRawAudioFromFile(fpath)

    # If no chunks, show error and skip
    if len(chunks) == 0:
        msg = 'Error: Cannot open audio file {}'.format(fpath)
        print(msg, flush=True)
        analyze.writeErrorLog(msg)
        return

    # Process each chunk
    try:
        start, end = 0, cfg.SIG_LENGTH
        results = {}
        samples = []
        timestamps = []
        for c in range(len(chunks)):

            # Add to batch
            samples.append(chunks[c])
            timestamps.append([start, end])

            # Advance start and end
            start += cfg.SIG_LENGTH - cfg.SIG_OVERLAP
            end = start + cfg.SIG_LENGTH

            # Check if batch is full or last chunk        
            if len(samples) < cfg.BATCH_SIZE and c < len(chunks) - 1:
                continue
            
            # Prepare sample and pass through model
            data = np.array(samples, dtype='float32')
            e = model.embeddings(data) 

            # Add to results
            for i in range(len(samples)):

                # Get timestamp
                s_start, s_end = timestamps[i]

                # Get prediction
                embeddings = e[i]       

                # Store embeddings
                results[str(s_start) + '-' + str(s_end)] = embeddings
            
            # Reset batch
            samples = []
            timestamps = []

    except:

        # Print traceback
        print(traceback.format_exc(), flush=True)

        # Write error log
        msg = 'Error: Cannot analyze audio file {}.\n{}'.format(fpath, traceback.format_exc())
        print(msg, flush=True)
        writeErrorLog(msg)
        return   

    # Save as embeddings file
    try:

        # We have to check if output path is a file or directory
        if not cfg.OUTPUT_PATH.rsplit('.', 1)[-1].lower() in ['txt', 'csv']:           

            fpath = fpath.replace(cfg.INPUT_PATH, '')
            fpath = fpath[1:] if fpath[0] in ['/', '\\'] else fpath
                        
            # Make target directory if it doesn't exist
            fdir = os.path.join(cfg.OUTPUT_PATH, os.path.dirname(fpath))
            if not os.path.exists(fdir):
                os.makedirs(fdir, exist_ok=True)

            saveAsEmbeddingsFile(results, os.path.join(cfg.OUTPUT_PATH, fpath.rsplit('.', 1)[0] + '.birdnet.embeddings.txt'))
        else:
            saveAsEmbeddingsFile(results, cfg.OUTPUT_PATH)
    except:

        # Print traceback
        print(traceback.format_exc(), flush=True)

        # Write error log
        msg = 'Error: Cannot save embeddings for {}.\n{}'.format(fpath, traceback.format_exc())
        print(msg, flush=True)
        writeErrorLog(msg)
        return

    delta_time = (datetime.datetime.now() - start_time).total_seconds()
    print('Finished {} in {:.2f} seconds'.format(fpath, delta_time), flush=True)

if __name__ == '__main__':

    # Clear error log
    #clearErrorLog() 

    # Parse arguments
    parser = argparse.ArgumentParser(description='Analyze audio files with BirdNET')
    parser.add_argument('--i', default='example/', help='Path to input file or folder. If this is a file, --o needs to be a file too.')
    parser.add_argument('--o', default='example/', help='Path to output file or folder. If this is a file, --i needs to be a file too.')
    parser.add_argument('--overlap', type=float, default=0.0, help='Overlap of prediction segments. Values in [0.0, 2.9]. Defaults to 0.0.')
    parser.add_argument('--threads', type=int, default=4, help='Number of CPU threads.')
    parser.add_argument('--batchsize', type=int, default=1, help='Number of samples to process at the same time. Defaults to 1.')

    args = parser.parse_args()

    # Set paths relative to script path (requested in #3)
    cfg.MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), cfg.MODEL_PATH)
    cfg.ERROR_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), cfg.ERROR_LOG_FILE)

    ### Make sure to comment out appropriately if you are not using args. ###

    # Set input and output path    
    cfg.INPUT_PATH = args.i
    cfg.OUTPUT_PATH = args.o

    # Parse input files
    if os.path.isdir(cfg.INPUT_PATH):
        cfg.FILE_LIST = analyze.parseInputFiles(cfg.INPUT_PATH)  
    else:
        cfg.FILE_LIST = [cfg.INPUT_PATH]

    # Set overlap
    cfg.SIG_OVERLAP = max(0.0, min(2.9, float(args.overlap)))

    # Set number of threads
    if os.path.isdir(cfg.INPUT_PATH):
        cfg.CPU_THREADS = int(args.threads)
        cfg.TFLITE_THREADS = 1
    else:
        cfg.CPU_THREADS = 1
        cfg.TFLITE_THREADS = int(args.threads)

    # Set batch size
    cfg.BATCH_SIZE = max(1, int(args.batchsize))

    # Add config items to each file list entry.
    # We have to do this for Windows which does not
    # support fork() and thus each process has to
    # have its own config. USE LINUX!
    flist = []
    for f in cfg.FILE_LIST:
        flist.append((f, cfg.getConfig()))

    # Analyze files   
    if cfg.CPU_THREADS < 2:
        for entry in flist:
            analyzeFile(entry)
    else:
        with Pool(cfg.CPU_THREADS) as p:
            p.map(analyzeFile, flist)

    # A few examples to test
    # python3 embeddings.py --i example/ --o example/ --threads 4
    # python3 embeddings.py --i example/soundscape.wav --o example/soundscape.birdnet.embeddings.txt --threads 4