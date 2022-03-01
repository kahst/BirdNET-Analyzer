import os
import argparse
import datetime

from multiprocessing import Pool

import config as cfg
import analyze
import model

def saveAsEmbeddingsFile(results, fpath):

    # Write embeddings to file
    with open(fpath, 'w') as f:
        for timestamp in results:
            f.write(timestamp.replace('-', '\t') + '\t' + ','.join(map(str, results[timestamp])) + '\n')

def analyzeFile(entry):

    # Get file path and restore cfg
    fpath = entry[0]
    cfg.INPUT_PATH = entry[1]
    cfg.OUTPUT_PATH = entry[2]

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
    start, end = 0, cfg.SIG_LENGTH
    results = {}
    for chunk in chunks:
        
        # Prepare sample and pass through model
        data = model.makeSample(chunk)
        embeddings = model.predict(data)        

        # Store top 5 results and advance indicies
        results[str(start) + '-' + str(end)] = embeddings
        start += cfg.SIG_LENGTH - cfg.SIG_OVERLAP
        end = start + cfg.SIG_LENGTH

    # Save as selection table
    if os.path.isdir(cfg.OUTPUT_PATH):
        fpath = fpath.replace(cfg.INPUT_PATH, '')
        fpath = fpath[1:] if fpath[0] in ['/', '\\'] else fpath
        saveAsEmbeddingsFile(results, os.path.join(cfg.OUTPUT_PATH, fpath.rsplit('.', 1)[0] + '.birdnet.embeddings.txt'))
    else:
        saveAsEmbeddingsFile(results, cfg.OUTPUT_PATH)

    delta_time = (datetime.datetime.now() - start_time).total_seconds()
    print('Finished {} in {:.2f} seconds'.format(fpath, delta_time), flush=True)

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Analyze audio files with BirdNET')
    parser.add_argument('--i', default='example/', help='Path to input file or folder. If this is a file, --o needs to be a file too.')
    parser.add_argument('--o', default='example/', help='Path to output file or folder. If this is a file, --i needs to be a file too.')
    parser.add_argument('--threads', type=int, default=4, help='Number of CPU threads.')

    args = parser.parse_args()

    ### Make sure to comment out appropriately if you are not using args. ###

    # Set input and output path    
    cfg.INPUT_PATH = args.i
    cfg.OUTPUT_PATH = args.o

    # Parse input files
    if os.path.isdir(cfg.INPUT_PATH):
        cfg.FILE_LIST = analyze.parseInputFiles(cfg.INPUT_PATH)  
    else:
        cfg.FILE_LIST = [cfg.INPUT_PATH]

    # Set number of threads
    if os.path.isdir(cfg.INPUT_PATH):
        cfg.CPU_THREADS = int(args.threads)
        cfg.TFLITE_THREADS = 1
    else:
        cfg.CPU_THREADS = 1
        cfg.TFLITE_THREADS = int(args.threads)

    # Add config items to each file list entry.
    # We have to do this for Windows which does not
    # support fork() and thus each process has to
    # have its own config. USE LINUX!
    flist = []
    for f in cfg.FILE_LIST:
        flist.append((f, cfg.INPUT_PATH, cfg.OUTPUT_PATH))

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