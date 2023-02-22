import os
import json
import bottle
import argparse
from datetime import datetime, date
import traceback
import tempfile

from multiprocessing import freeze_support

import config as cfg
import analyze

def clearErrorLog():

    if os.path.isfile(cfg.ERROR_LOG_FILE):
        os.remove(cfg.ERROR_LOG_FILE)

def writeErrorLog(msg):

    with open(cfg.ERROR_LOG_FILE, 'a') as elog:
        elog.write(msg + '\n')

def resultPooling(lines, num_results=5, pmode='avg'):

    # Parse results
    results = {}
    for line in lines:
        d = line.split('\t')
        species = d[2].replace(', ', '_')
        score = float(d[-1])
        if not species in results:
            results[species] = []
        results[species].append(score)

    # Compute score for each species
    for species in results:

        if pmode == 'max':
            results[species] = max(results[species])
        else:
            results[species] = sum(results[species]) / len(results[species])

    # Sort results
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    return results[:num_results]

@bottle.route('/healthcheck', method='GET')
def healthcheck():
    data = {'msg': 'Server is healthy.'}
    return json.dumps(data)

@bottle.route('/analyze', method='POST')
def handleRequest():

    # Print divider
    print('{}  {}  {}'.format('#' * 20, datetime.now(), '#' * 20))

    # Get request payload
    upload = bottle.request.files.get('audio')
    mdata = json.loads(bottle.request.forms.get('meta'))
    print(mdata)

    # Get filename
    name, ext = os.path.splitext(upload.filename.lower())

    file_path_tmp = None

    # Save file
    try:
        if ext.lower() in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
            if 'save' in mdata and mdata['save']:
                save_path = os.path.join(cfg.FILE_STORAGE_PATH, str(date.today()))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                file_path = os.path.join(save_path, name + ext)
            else:
                save_path = ''
                file_path_tmp = tempfile.NamedTemporaryFile(suffix=ext.lower(), delete=False)
                file_path_tmp.close()
                file_path = file_path_tmp.name
            upload.save(file_path, overwrite=True)

        else:
            data = {'msg': 'Filetype not supported.'}
            return json.dumps(data)
    
    except:
        if file_path_tmp is not None:
            os.unlink(file_path_tmp.name)

        # Print traceback
        print(traceback.format_exc(), flush=True)

        # Write error log
        msg = 'Error: Cannot save file {}.\n{}'.format(file_path, traceback.format_exc())
        print(msg, flush=True)
        writeErrorLog(msg)

        # Return error
        data = {'msg': 'Error while saving file.'}
        return json.dumps(data)

    # Analyze file
    try:
        
        # Set config based on mdata
        if 'lat' in mdata and 'lon' in mdata:
            cfg.LATITUDE = float(mdata['lat'])
            cfg.LONGITUDE = float(mdata['lon'])
        else:
            cfg.LATITUDE = -1
            cfg.LONGITUDE = -1
        if 'week' in mdata:
            cfg.WEEK = int(mdata['week'])
        else:
            cfg.WEEK = -1
        if 'overlap' in mdata:
            cfg.SIG_OVERLAP = max(0.0, min(2.9, float(mdata['overlap'])))
        else:
            cfg.SIG_OVERLAP = 0.0
        if 'sensitivity' in mdata:
            cfg.SIGMOID_SENSITIVITY = max(0.5, min(1.0 - (float(mdata['sensitivity']) - 1.0), 1.5))
        else:
            cfg.SIGMOID_SENSITIVITY = 1.0
        if 'sf_thresh' in mdata:
            cfg.LOCATION_FILTER_THRESHOLD = max(0.01, min(0.99, float(mdata['sf_thresh'])))
        else:
            cfg.LOCATION_FILTER_THRESHOLD = 0.03       

        # Set species list
        if not cfg.LATITUDE == -1 and not cfg.LONGITUDE == -1:
            analyze.predictSpeciesList() 

        # Analyze file
        success = analyze.analyzeFile((file_path, cfg.getConfig()))

        # Parse results
        if success:
            
            # Open result file
            lines = []
            with open(cfg.OUTPUT_PATH, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    lines.append(line.strip())

            # Pool results
            if 'pmode' in mdata and mdata['pmode'] in ['avg', 'max']:
                pmode = mdata['pmode']
            else:
                pmode = 'avg'
            if 'num_results' in mdata:
                num_results = min(99, max(1, int(mdata['num_results'])))
            else:
                num_results = 5
            results = resultPooling(lines, num_results, pmode)

            # Prepare response
            data = {'msg': 'success', 'results': results, 'meta': mdata}

            # Save response as metadata file
            if 'save' in mdata and mdata['save']:
                with open(file_path.rsplit('.', 1)[0] + '.json', 'w') as f:
                    json.dump(data, f, indent=2)

            # Return response
            del data['meta']
            return json.dumps(data)

        else:
            data = {'msg': 'Error during analysis.'}
            return json.dumps(data)

    except Exception as e:

        # Print traceback
        print(traceback.format_exc(), flush=True)

        # Write error log
        msg = 'Error: Cannot analyze file {}.\n{}'.format(file_path, traceback.format_exc())
        print(msg, flush=True)
        writeErrorLog(msg)

        data = {'msg': 'Error during analysis: {}'.format(str(e))}      
        return json.dumps(data)    
    finally:
        if file_path_tmp is not None:
            os.unlink(file_path_tmp.name)

if __name__ == '__main__':

    # Freeze support for excecutable
    freeze_support()

    # Clear error log
    clearErrorLog()

    # Parse arguments
    parser = argparse.ArgumentParser(description='API endpoint server to analyze files remotely.')
    parser.add_argument('--host', default='0.0.0.0', help='Host name or IP address of API endpoint server. Defaults to \'0.0.0.0\'')   
    parser.add_argument('--port', type=int, default=8080, help='Port of API endpoint server. Defaults to 8080.')   
    parser.add_argument('--spath', default='uploads/', help='Path to folder where uploaded files should be stored. Defaults to \'/uploads\'.')
    parser.add_argument('--threads', type=int, default=4, help='Number of CPU threads for analysis. Defaults to 4.')
    parser.add_argument('--locale', default='en', help='Locale for translated species common names. Values in [\'af\', \'de\', \'it\', ...] Defaults to \'en\'.')

    args = parser.parse_args()

   # Load eBird codes, labels
    cfg.CODES = analyze.loadCodes()
    cfg.LABELS = analyze.loadLabels(cfg.LABELS_FILE)

    # Load translated labels
    lfile = os.path.join(cfg.TRANSLATED_LABELS_PATH, os.path.basename(cfg.LABELS_FILE).replace('.txt', '_{}.txt'.format(args.locale)))
    if not args.locale in ['en'] and os.path.isfile(lfile):
        cfg.TRANSLATED_LABELS = analyze.loadLabels(lfile)
    else:
        cfg.TRANSLATED_LABELS = cfg.LABELS  

    # Set storage file path
    cfg.FILE_STORAGE_PATH = args.spath

    # Set min_conf to 0.0, because we want all results
    cfg.MIN_CONFIDENCE = 0.0

    output_file = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
    output_file.close()

    # Set path for temporary result file
    cfg.OUTPUT_PATH = output_file.name

    # Set result type
    cfg.RESULT_TYPE = 'audacity'

    # Set number of TFLite threads
    cfg.TFLITE_THREADS = max(1, int(args.threads))

    # Run server
    print('UP AND RUNNING! LISTENING ON {}:{}'.format(args.host, args.port), flush=True)
    try:
        bottle.run(host=args.host, port=args.port, quiet=True)
    finally:
        os.unlink(output_file.name)
