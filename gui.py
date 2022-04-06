import os
import time
import traceback
import webview

from multiprocessing import Pool, freeze_support

import config as cfg
import analyze
import model

def registerWindow(window):

    # Save as global variable
    global WINDOW
    WINDOW = window

    # Expose functions
    WINDOW.expose(runAnalysis)  
    WINDOW.expose(openFolderDialog)  
    WINDOW.expose(openFileDialog)  

def openFolderDialog():
    path = WINDOW.create_file_dialog(dialog_type=webview.FOLDER_DIALOG, directory='', allow_multiple=False)
    return path

def openFileDialog():
    path = WINDOW.create_file_dialog(dialog_type=webview.OPEN_DIALOG, directory='', allow_multiple=False)
    return path

def show(msg):
    WINDOW.evaluate_js('showStatus("' + str(msg) + '")')

def runAnalysis(config):

    try:

        # Load eBird codes, labels
        cfg.CODES = analyze.loadCodes()
        cfg.LABELS = analyze.loadLabels(cfg.LABELS_FILE)

        # Load translated labels
        lfile = os.path.join(cfg.TRANSLATED_LABELS_PATH, os.path.basename(cfg.LABELS_FILE).replace('.txt', '_{}.txt'.format(config['locale'])))
        if not config['locale'] in ['en'] and os.path.isfile(lfile):
            cfg.TRANSLATED_LABELS = analyze.loadLabels(lfile)
        else:
            cfg.TRANSLATED_LABELS = cfg.LABELS  

        # Generate species list
        if len(config['slist_path']) == 0:
            cfg.SPECIES_LIST_FILE = None
        else:
            cfg.SPECIES_LIST_FILE = config['slist_path']
        cfg.SPECIES_LIST = analyze.loadSpeciesList(cfg.SPECIES_LIST_FILE)

        cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK = float(config['lat']), float(config['lon']), int(config['week'])
        if not cfg.LATITUDE == -1 and not cfg.LONGITUDE == -1:
            analyze.predictSpeciesList()
        if len(cfg.SPECIES_LIST) == 0:
            show('Species list contains {} species'.format(len(cfg.LABELS)))
        else:        
            show('Species list contains {} species'.format(len(cfg.SPECIES_LIST)))

        # Set input and output path    
        cfg.INPUT_PATH = config['input_path']
        cfg.OUTPUT_PATH = config['output_path']

        # Parse input files
        if os.path.isdir(cfg.INPUT_PATH):
            cfg.FILE_LIST = analyze.parseInputFiles(cfg.INPUT_PATH)  
        else:
            cfg.FILE_LIST = [cfg.INPUT_PATH]

        # Set confidence threshold
        cfg.MIN_CONFIDENCE = max(0.01, min(0.99, float(config['min_conf'])))

        # Set sensitivity
        cfg.SIGMOID_SENSITIVITY = max(0.5, min(1.0 - (float(config['sensitivity']) - 1.0), 1.5))

        # Set overlap
        cfg.SIG_OVERLAP = max(0.0, min(2.9, float(config['overlap'])))

        # Set result type
        cfg.RESULT_TYPE = config['rtype'].lower()

        # Set number of threads
        cfg.TFLITE_THREADS = int(config['threads'])

        # Process files
        for f in cfg.FILE_LIST:
            p_start = time.time()
            show('Processing file {}'.format(f.replace(os.sep, '/').replace('//', '/')))
            analyze.analyzeFile((f, cfg.getConfig()))   
            show('Finished {} in {:.2f} seconds'.format(f.replace(os.sep, '/').replace('//', '/'), time.time() - p_start))     

    except:
        traceback.print_exc()
        show(traceback.format_exc())

    return 'Analysis done!'

if __name__ == '__main__':

    freeze_support()

    window = webview.create_window('BirdNET-Analyzer', 'gui/index.html', width=1024, height=960)
    webview.start(registerWindow, window, debug=True)