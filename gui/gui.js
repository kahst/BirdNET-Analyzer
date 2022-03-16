window.addEventListener('pywebviewready', function() {
        

})

function showStatus(msg) {
    var statustext = document.getElementById('status')

    statustext.value += msg + '\n'

    statustext.scrollTop = statustext.scrollHeight 
}

function analyze() {
    var inputPath = document.getElementById('input-path').value
    var outputPath = document.getElementById('output-path').value
    var slistPath = document.getElementById('slist-path').value
    var lat = document.getElementById('latitude').value
    var lon = document.getElementById('longitude').value
    var week = document.getElementById('week').value
    var overlap = document.getElementById('overlap').value
    var sensitivity = document.getElementById('sensitivity').value
    var min_conf = document.getElementById('minconf').value
    var threads = document.getElementById('threads').value
    var locale = document.getElementById('locale').value
    var rtype = document.getElementById('rtype').value

    config = {
        input_path: inputPath,
        output_path: outputPath,
        slist_path: slistPath,
        lat: lat,
        lon: lon,
        week: week,
        overlap: overlap,
        sensitivity: sensitivity,
        min_conf: min_conf,
        threads: threads,
        locale: locale,
        rtype: rtype
    }

    pywebview.api.runAnalysis(config).then(function(msg) {
        showStatus(msg)
    })

    showStatus('Starting analysis...')
    
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