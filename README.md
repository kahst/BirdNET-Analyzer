# PRELIMINARY MODEL ONLY, WILL TRAIN BETTER VERSION

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

# BirdNET-Analyzer
Vanilla BirdNET analyzer for quick testing.

# Setup (Ubuntu)

Install Python 3:
```
sudo apt-get update
sudo apt-get install python3-dev python3-pip
sudo pip3 install --upgrade pip
```

Install Tensorflow (has to be 2.5 or later):
```
sudo pip3 install tensorflow
```

Install Librosa to handle audio files:

```
sudo pip3 install librosa
sudo apt-get install ffmpeg
```

Clone the repository

```
git clone https://github.com/kahst/BirdNET-Analyzer.git
cd BirdNET-Analyzer
```

# Usage

1. Inspect config file for options and settings, especially inference settings. Specify a custom species list if needed and adjust the number of threads TFLite can use to run the inference.

2. Run `analyzer.py` to analyze an audio file. You need to set paths for the audio file and selection table output. Here is an example:

```
# Analyze a file
result = analyzeFile('example/soundscape.wav')

# Generate Raven selection table
saveAsSelectionTable(result, 'example/soundscape_selections.txt')
```

3. This is a very basic version of the analysis workflow, you might need to adjust it to your own needs.

4. Please open an issue to ask for new features or to document unexpected behavior.

5. I will keep models up to date and upload new checkpoints whenever there is an improvement in performance. I will also provide quantized and pruned model files for distribution.