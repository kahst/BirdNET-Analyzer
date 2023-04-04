import os
import argparse
import traceback

import numpy as np
from multiprocessing import Pool

import config as cfg
import audio

# Set numpy random seed
np.random.seed(cfg.RANDOM_SEED)

def clearErrorLog():

    if os.path.isfile(cfg.ERROR_LOG_FILE):
        os.remove(cfg.ERROR_LOG_FILE)

def writeErrorLog(msg):

    with open(cfg.ERROR_LOG_FILE, 'a') as elog:
        elog.write(str(msg) + '\n')

def detectRType(line):

    if line.lower().startswith('selection'):
        return 'table'
    elif line.lower().startswith('filepath'):
        return 'r'
    elif line.lower().startswith('indir'):
        return 'kaleidoscope'
    elif line.lower().startswith('start (s)'):
        return 'csv'
    else:
        return 'audacity'

def parseFolders(apath, rpath, allowed_filetypes={'audio': ['wav', 'flac', 'mp3', 'ogg', 'm4a'], 'results': ['txt', 'csv']}):

    data = {}
    # Get all audio files
    for root, dirs, files in os.walk(apath):
        for f in files:
            if f.split('.')[-1].lower() in allowed_filetypes['audio']:
                data[f.rsplit('.', 1)[0]] = {'audio': os.path.join(root, f), 'result': ''}

    # Get all result files
    for root, dirs, files in os.walk(rpath):
        for f in files:
            if f.split('.')[-1] in allowed_filetypes['results'] and f.find('.BirdNET.') != -1:
                data[f.split('.BirdNET.')[0]]['result'] = os.path.join(root, f)

    # Convert to list
    flist = []
    for f in data:
        if len(data[f]['result']) > 0:
            flist.append(data[f])

    print('Found {} audio files with valid result file.'.format(len(flist)))

    return flist

def parseFiles(flist, max_segments=100):

    species_segments = {}
    for f in flist:

        # Paths
        afile = f['audio'] 
        rfile = f['result']

        # Get all segments for result file
        segments = findSegments(afile, rfile)

        # Parse segments by species
        for s in segments:
            if s['species'] not in species_segments:
                species_segments[s['species']] = []
            species_segments[s['species']].append(s)

    # Shuffle segments for each species and limit to max_segments
    for s in species_segments:
        np.random.shuffle(species_segments[s])
        species_segments[s] = species_segments[s][:max_segments]

    # Make dict of segments per audio file
    segments = {}
    seg_cnt = 0
    for s in species_segments:
        for seg in species_segments[s]:
            if not seg['audio'] in segments:
                segments[seg['audio']] = []
            segments[seg['audio']].append(seg)
            seg_cnt += 1

    print('Found {} segments in {} audio files.'.format(seg_cnt, len(segments)))

    # Convert to list
    flist = []
    for f in segments:
        flist.append((f, segments[f]))

    return flist

def findSegments(afile, rfile):

    segments = []

    # Open and parse result file
    lines = []
    with open(rfile, 'r', encoding='utf-8') as rf:
        for line in rf.readlines():
            lines.append(line.strip())

    # Auto-detect result type
    rtype = detectRType(lines[0])

    # Get start and end times based on rtype
    confidence = 0
    for i in range(len(lines)):
        if rtype == 'table' and i > 0:
            d = lines[i].split('\t')
            start = float(d[3])
            end = float(d[4])
            species = d[-2]
            confidence = float(d[-1])

        elif rtype == 'audacity':
            d = lines[i].split('\t')
            start = float(d[0])
            end = float(d[1])
            species = d[2].split(', ')[1]
            confidence = float(d[-1])

        elif rtype == 'r' and i > 0:
            d = lines[i].split(',')
            start = float(d[1])
            end = float(d[2])
            species = d[4]
            confidence = float(d[5])

        elif rtype == 'kaleidoscope' and i > 0:
            d = lines[i].split(',')
            start = float(d[3])
            end = float(d[4]) + start
            species = d[5]
            confidence = float(d[7])

        elif rtype == 'csv' and i > 0:
            d = lines[i].split(',')
            start = float(d[0])
            end = float(d[1])
            species = d[3]
            confidence = float(d[4])

        # Check if confidence is high enough
        if confidence >= cfg.MIN_CONFIDENCE:
            segments.append({'audio': afile, 'start': start, 'end': end, 'species': species, 'confidence': confidence})

    return segments

def extractSegments(item):

    # Paths and config
    afile = item[0][0]
    segments = item[0][1]
    seg_length = item[1]
    cfg.setConfig(item[2])

    # Status
    print('Extracting segments from {}'.format(afile))

    try:
        # Open audio file
        sig, _ = audio.openAudioFile(afile, cfg.SAMPLE_RATE)
    except Exception as ex:
        print('Error: Cannot open audio file {}'.format(afile), flush=True)
        writeErrorLog(ex)
        return

    # Extract segments
    seg_cnt = 1
    for seg in segments:

        try:
            
            # Get start and end times
            start = int(seg['start'] * cfg.SAMPLE_RATE)
            end = int(seg['end'] * cfg.SAMPLE_RATE)
            offset = ((seg_length * cfg.SAMPLE_RATE) - (end - start)) // 2
            start = max(0, start - offset)
            end = min(len(sig), end + offset)  

            # Make sure segmengt is long enough
            if end > start:

                # Get segment raw audio from signal
                seg_sig = sig[int(start):int(end)]

                # Make output path
                outpath = os.path.join(cfg.OUTPUT_PATH, seg['species'])
                if not os.path.exists(outpath):
                    os.makedirs(outpath, exist_ok=True)

                # Save segment
                seg_name = '{:.3f}_{}_{}.wav'.format(seg['confidence'], seg_cnt, seg['audio'].split(os.sep)[-1].rsplit('.', 1)[0])
                seg_path = os.path.join(outpath, seg_name)
                audio.saveSignal(seg_sig, seg_path)
                seg_cnt += 1

        except:

            # Print traceback
            print(traceback.format_exc(), flush=True)

            # Write error log
            msg = 'Error: Cannot extract segments from {}.\n{}'.format(afile, traceback.format_exc())
            print(msg, flush=True)
            writeErrorLog(msg)
            break

if __name__ == '__main__':

    # Clear error log
    #clearErrorLog()

    # Parse arguments
    parser = argparse.ArgumentParser(description='Extract segments from audio files based on BirdNET detections.')
    parser.add_argument('--audio', default='example/', help='Path to folder containing audio files.')
    parser.add_argument('--results', default='example/', help='Path to folder containing result files.')
    parser.add_argument('--o', default='example/', help='Output folder path for extracted segments.')
    parser.add_argument('--min_conf', type=float, default=0.1, help='Minimum confidence threshold. Values in [0.01, 0.99]. Defaults to 0.1.')
    parser.add_argument('--max_segments', type=int, default=100, help='Number of randomly extracted segments per species.')
    parser.add_argument('--seg_length', type=float, default=3.0, help='Length of extracted segments in seconds. Defaults to 3.0.')
    parser.add_argument('--threads', type=int, default=4, help='Number of CPU threads.')

    args = parser.parse_args()

    # Parse audio and result folders
    cfg.FILE_LIST = parseFolders(args.audio, args.results)
    
    # Set output folder
    cfg.OUTPUT_PATH = args.o

    # Set number of threads
    cfg.CPU_THREADS = int(args.threads)

    # Set confidence threshold
    cfg.MIN_CONFIDENCE = max(0.01, min(0.99, float(args.min_conf)))

    # Parse file list and make list of segments
    cfg.FILE_LIST = parseFiles(cfg.FILE_LIST, max(1, int(args.max_segments)))

    # Add config items to each file list entry.
    # We have to do this for Windows which does not
    # support fork() and thus each process has to
    # have its own config. USE LINUX!
    flist = []
    for entry in cfg.FILE_LIST:
        flist.append((entry, max(cfg.SIG_LENGTH, float(args.seg_length)), cfg.getConfig()))
    
    # Extract segments   
    if cfg.CPU_THREADS < 2:
        for entry in flist:
            extractSegments(entry)
    else:
        with Pool(cfg.CPU_THREADS) as p:
            p.map(extractSegments, flist)

    # A few examples to test
    # python3 segments.py --audio example/ --results example/ --o example/segments/ 
    # python3 segments.py --audio example/ --results example/ --o example/segments/ --seg_length 5.0 --min_conf 0.1 --max_segments 100 --threads 4
