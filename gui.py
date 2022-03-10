import webview

import analyze

html = """
<!DOCTYPE html>
<html>
<head lang="en">
<meta charset="UTF-8">

<style>
    #response-container {
        display: none;
        padding: 3rem;
        margin: 3rem 5rem;
        font-size: 120%;
        border: 5px dashed #ccc;
    }

    label {
        margin-left: 0.3rem;
        margin-right: 0.3rem;
    }

    button {
        font-size: 100%;
        padding: 0.5rem;
        margin: 0.3rem;
        text-transform: uppercase;
    }

</style>
</head>
<body>


<h1>BirdNET Analyzer</h1>
<input type="text" id="input-path" placeholder="Input path" value="" size=50>
<button onClick="openFolderDialog('input-path')">Select input path</button><br/>
<input type="text" id="output-path" placeholder="Output path" value="" size=50>
<button onClick="openFolderDialog('output-path')">Select output path</button><br/>
<input type="text" id="slist-path" placeholder="Species list path" value="" size=50>
<button onClick="openFileDialog('slist-path')">Select species list</button><br/>

<button id="analysis-btn" onClick="analyze()">Start analysis</button><br/>
<div id="response-container"></div>
<script>
    window.addEventListener('pywebviewready', function() {
        

    })

    function showResponse(response) {
        var container = document.getElementById('response-container')

        container.innerText = response.message
        container.style.display = 'block'
    }

    function analyze() {
        var btn = document.getElementById('analysis-btn')
        var inputPath = document.getElementById('input-path').value
        var outputPath = document.getElementById('output-path').value
        var slistPath = document.getElementById('slist-path').value

        pywebview.api.runAnalysis(inputPath, outputPath, slistPath).then(function(response) {
            showResponse(response)
        })

        showResponse({message: 'Starting analysis...'})
        
    }

    function openFolderDialog(elementId) {
        pywebview.api.openFolderDialog().then(function(path) {
            document.getElementById(elementId).value = path;
        })
    }

    function openFileDialog(elementId) {
        pywebview.api.openFileDialog().then(function(path) {
            document.getElementById(elementId).value = path;
        })
    }

</script>
</body>
</html>
"""

def runAnalysis(in_path, out_path, slist_path):

    # Load eBird codes, labels
    CODES = analyze.loadCodes()
    LABELS = analyze.loadLabels()

    SPECIES_LIST = analyze.loadSpeciesList(slist_path)

    FLIST = analyze.parseInputFiles(in_path)
    for k in range(50):
        for i in range(len(FLIST)):
            analyze.analyzeFile((FLIST[i], CODES, LABELS, SPECIES_LIST, in_path, out_path, 0.1, 1.0, 'table'))

            WINDOW.evaluate_js('showResponse({message: "Progress: ' + str(k) + '"})')
    
    response = {
        'message': 'Analysis done'
    }
    return response

def openFolderDialog():
    path = WINDOW.create_file_dialog(dialog_type=webview.FOLDER_DIALOG, directory='', allow_multiple=False)
    return path

def openFileDialog():
    path = WINDOW.create_file_dialog(dialog_type=webview.OPEN_DIALOG, directory='', allow_multiple=False)
    return path

def registerWindow(window):

    # Save as global variable
    global WINDOW
    WINDOW = window

    # Expose functions
    WINDOW.expose(runAnalysis)  
    WINDOW.expose(openFolderDialog)  
    WINDOW.expose(openFileDialog)  

if __name__ == '__main__':
    window = webview.create_window('API example', html=html)
    webview.start(registerWindow, window, debug=True)