# tests/test_analyze__main.py
#
# These are end-to-end tests that run module `birdnet_analyzer.analyze`
# as if calling it from the command-line.
#
# When run with certain arguments on known example input,
# an output file should be generated with expected specific format and data
# as found in tests/resources/SNAPSHOT.analyze__main.*)

import subprocess
import os

def test_analyze_case1_example_min_conf(tmp_path):
    """
    Tests a command similar to:
    python3 analyze.py --i example/ --o example/ --slist example/ --min_conf 0.5 --threads 4
    """
    # GIVEN: analyze is called on example/ dir with 1 sound file, soundscape.wav, with a minimum confidence level of 0.5
    output_dir = tmp_path
    # comment the following line in in order to write a file to troubleshoot or take a new snapshot
    # output_dir = 'birdnet_analyzer/example'
    cmd = [
        'python3',
        '-m','birdnet_analyzer.analyze',
        '--i', 'birdnet_analyzer/example/',
        '--o', output_dir,
        '--slist', 'example/',
        '--min_conf', '0.5',
        '--threads', '4'
    ]

    # WHEN: analyze is run
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # THEN: the file created by analyze matches the expected SNAPSHOT (e.g. with fewer results than default confidence threshold)
    SNAPSHOT_FILE_PATH='tests/resources/SNAPSHOT.analyze__main.case1.expected.soundscape.table.txt'
    output_file = f'{output_dir}/soundscape.BirdNET.selection.table.txt'
    assert result.returncode == 0, f"Command failed with error: {result.stderr}"
    assert os.path.exists(output_file), "File should exist but doesn't: <<file_name_minus_extension>>.BirdNET.selection.table.txt"
    with open(output_file, 'r') as f1, open(SNAPSHOT_FILE_PATH, 'r') as f2:
        assert f1.read() == f2.read()


def test_analyze_case2_soundscape(tmp_path):
    """
    Tests a command similar to:
    python3 analyze.py --i example/soundscape.wav --o example/soundscape.BirdNET.selection.table.txt --slist example/species_list.txt --threads 8
    """
    
    # GIVEN: analyze is called on single file: soundscape.wav
    output_dir = tmp_path
    # comment the following line in in order to write a file to troubleshoot or take a new snapshot
    # output_dir = 'birdnet_analyzer/example'
    cmd = [
        'python3',
        '-m','birdnet_analyzer.analyze',
        '--i', 'birdnet_analyzer/example/soundscape.wav',
        '--o', output_dir,
        '--slist', 'example/species_list.txt',
        '--threads', '8'
    ]
    
    # WHEN: analyze is run
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # THEN: the file created by analyze matches the expected SNAPSHOT
    SNAPSHOT_FILE_PATH='tests/resources/SNAPSHOT.analyze__main.case2.expected.soundscape.table.txt'
    output_file = f'{output_dir}/soundscape.BirdNET.selection.table.txt'
    assert result.returncode == 0, f"Command failed with error: {result.stderr}"
    assert os.path.exists(output_file), "File should exist but doesn't: <<file_name_minus_extension>>.BirdNET.selection.table.txt"
    with open(output_file, 'r') as f1, open(SNAPSHOT_FILE_PATH, 'r') as f2:
        assert f1.read() == f2.read()


def test_analyze_case3_latlon_week4_sensitivity_rtype_de(tmp_path):
    """
    Tests a command similar to:
    python3 analyze.py --i example/ --o example/ --lat 42.5 --lon -76.45 --week 4 --sensitivity 1.0 --rtype table --locale de
    """

    # GIVEN: analyze is called on with lat, lon, week, sensitivity, rtype, and locale set
    output_dir = tmp_path
    # comment the following line in in order to write a file to troubleshoot or take a new snapshot
    # output_dir = 'birdnet_analyzer/example'
    cmd = [
        'python3',
        '-m','birdnet_analyzer.analyze',
        '--i', 'birdnet_analyzer/example/',
        '--o', output_dir,
        '--slist', 'example/',
        '--lat', '42.5',
        '--lon', '-76.45',
        '--week', '4',
        '--sensitivity', '1.0',
        '--rtype', 'table',
        '--locale', 'de'
    ]

    # WHEN: analyze is run
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # THEN: the file created by analyze matches the expected SNAPSHOT (e.g. in German with expected data)
    SNAPSHOT_FILE_PATH='tests/resources/SNAPSHOT.analyze__main.case3.expected.soundscape.table.txt'
    output_file = f'{output_dir}/soundscape.BirdNET.selection.table.txt'
    assert result.returncode == 0, f"Command failed with error: {result.stderr}"
    assert os.path.exists(output_file), "File should exist but doesn't: <<file_name_minus_extension>>.BirdNET.selection.table.txt"
    with open(output_file, 'r') as f1, open(SNAPSHOT_FILE_PATH, 'r') as f2:
        assert f1.read() == f2.read()
