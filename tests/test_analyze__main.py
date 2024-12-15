# tests/test_analyze.py
import subprocess
import os

def test_analyze_case1_soundscape_dirInput_listTxtInDir_min05_4threads(tmp_path):
    """
    python3 analyze.py --i example/ --o example/ --slist example/ --min_conf 0.5 --threads 4
    """
    # using tmp_path will automatically clean up the file written by this test
    output_dir = tmp_path
    # comment the following line in in order to write a file to troubleshoot or take a new snapshot
    # output_dir = 'birdnet_analyzer/example'
    output_file = f'{output_dir}/soundscape.BirdNET.selection.table.txt'
    cmd = [
        'python3',
        '-m','birdnet_analyzer.analyze',
        '--i', 'birdnet_analyzer/example/',
        '--o', output_dir,
        '--slist', 'example/',
        '--min_conf', '0.5',
        '--threads', '4'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Assertions
    assert result.returncode == 0, f"Command failed with error: {result.stderr}"
    # assert os.path.exists('birdnet_analyzer/example/soundscape.BirdNET.selection.table.txt')
    assert os.path.exists(output_file), "File should exist but doesn't: <<file_name_minus_extension>>.BirdNET.selection.table.txt"
    SNAPSHOT_FILE_PATH='tests/resources/SNAPSHOT.analyze__main.case1.expected.soundscape.BirdNET.selection.table.txt'
    with open(output_file, 'r') as f1, open(SNAPSHOT_FILE_PATH, 'r') as f2:
        assert f1.read() == f2.read()

def test_analyze_soundscape_fileInput_listTxt_8threads(tmp_path):
    """
    # python3 analyze.py --i example/soundscape.wav --o example/soundscape.BirdNET.selection.table.txt --slist example/species_list.txt --threads 8
    """
    # using tmp_path will automatically clean up the file written by this test
    output_dir = tmp_path
    # comment the following line in in order to write a file to troubleshoot or take a new snapshot
    # output_dir = 'birdnet_analyzer/example'
    output_file = f'{output_dir}/soundscape.BirdNET.selection.table.txt'
    
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
    assert os.path.exists(output_file), "File should exist but doesn't: <<file_name_minus_extension>>.BirdNET.selection.table.txt"

    SNAPSHOT_FILE_PATH='tests/resources/SNAPSHOT.analyze__main.case2.expected.soundscape.BirdNET.selection.table.txt'
    with open(output_file, 'r') as f1, open(SNAPSHOT_FILE_PATH, 'r') as f2:
        assert f1.read() == f2.read()

def test_analyze_latlon_week4_sensitivity_rtypeTable_de(tmp_path):
    """
    python3 analyze.py --i example/ --o example/ --lat 42.5 --lon -76.45 --week 4 --sensitivity 1.0 --rtype table --locale de
    """
    # using tmp_path will automatically clean up the file written by this test
    output_dir = tmp_path
    # comment the following line in in order to write a file to troubleshoot or take a new snapshot
    # output_dir = 'birdnet_analyzer/example'
    output_file = f'{output_dir}/soundscape.BirdNET.selection.table.txt'
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
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Assertions
    assert result.returncode == 0, f"Command failed with error: {result.stderr}"
    assert os.path.exists(output_file), "File should exist but doesn't: <<file_name_minus_extension>>.BirdNET.selection.table.txt"

    SNAPSHOT_FILE_PATH='tests/resources/SNAPSHOT.analyze__main.case3.esxpected.soundscape.BirdNET.selection.table.txt'
    with open(output_file, 'r') as f1, open(SNAPSHOT_FILE_PATH, 'r') as f2:
        assert f1.read() == f2.read()

        
def test_tmp_path_doesnt_keep_file(tmp_path):
    """
    For those not familiar with this "tmp_path" mechanism,
    this test is just to illustrate that tmp files do not
    persist, even between tests in the same file.
    """
    output_file = f'{tmp_path}/soundscape.BirdNET.selection.table.txt'
    assert not os.path.exists(output_file), 'File should not exist'
