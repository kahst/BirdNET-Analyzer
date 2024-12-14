# tests/test_analyze.py
import pytest
from birdnet_analyzer import analyze
import birdnet_analyzer.config as cfg
import os

@pytest.fixture
def sample_results():
    return {
        # "0-3": [("Species_A", 0.95), ("Species_B", 0.85)],
        # "3-6": [("Species_C", 0.75)]
    }

def test_get_sorted_timestamps():
    results = {
        "3-6": [],
        "0-3": [],
        "6-9": []
    }
    expected = ["0-3", "3-6", "6-9"]
    assert analyze.getSortedTimestamps(results) == expected

def test_generate_csv(tmp_path, sample_results):
    result_path = tmp_path / "test_output.csv"
    audio_path = "sample.wav"
    
    analyze.generate_csv(
        analyze.getSortedTimestamps(sample_results),
        sample_results,
        audio_path,
        str(result_path)
    )
    
    assert result_path.exists()
    content = result_path.read_text()
    assert "Start (s),End (s),Scientific name" in content
    # assert "Species_A" in content

# def test_analyzeFile():
#     cfg.INPUT_PATH = "/Users/ken/Documents/wk/BirdNET-Analyzer/birdnet_analyzer/example/"
#     assert analyze.analyzeFile(
#         ('/Users/ken/Documents/wk/BirdNET-Analyzer/birdnet_analyzer/example/soundscape.wav', cfg.getConfig())
#         )
    

def test_getRawAudioFromFile():
    ## with known test file at sample rate 48000
    # starting from offset: 0, duration: 600
    chunks = analyze.getRawAudioFromFile('birdnet_analyzer/example/soundscape.wav', 0, 600)
    assert len(chunks) == 40, "expected: 40 chunks"
    assert chunks[0].shape == (144000,), "chunk has shape of a single 1-D row of 144,000 items"
    chunks = analyze.getRawAudioFromFile('birdnet_analyzer/example/soundscape.wav', 600, 600)
    assert len(chunks) == 0, "expected: 0 chunks. Our sample has only 1 set of chunks."

import subprocess
import os

def test_analyze_main(tmp_path):
    # Create output path in temporary directory
    # output_file = str(tmp_path / "soundscape.BirdNET.selection.table.txt")
    output_dir = 'birdnet_analyzer/example'
    output_file = 'birdnet_analyzer/example/soundscape.BirdNET.selection.table.txt'
    # Prepare the command line arguments
    cmd = [
        'python3',
        '-m','birdnet_analyzer.analyze',
        '--i', 'birdnet_analyzer/example/soundscape.wav',
        '--o', output_dir,
        '--slist', 'example/species_list.txt',
        '--threads', '8'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Assertions
    assert result.returncode == 0, f"Command failed with error: {result.stderr}"
    # assert os.path.exists('birdnet_analyzer/example/soundscape.BirdNET.selection.table.txt')
    assert os.path.exists(output_file)

    with open(output_file, 'r') as f1, open('birdnet_analyzer/example/bk.soundscape.BirdNET.selection.table.txt', 'r') as f2:
        assert f1.read() == f2.read()