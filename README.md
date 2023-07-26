//***********************************************
//***************** SETTINGS ********************
//***********************************************

:doctype: book
:use-link-attrs:
:linkattrs:

// Github Icons
ifdef::env-github[]
:tip-caption: :bulb:
:note-caption: :information_source:
:important-caption: :heavy_exclamation_mark:
:caution-caption: :fire:
:warning-caption: :warning:
endif::[]

// Table of Contents
:toc:
:toclevels: 2
:toc-title: 
:toc-placement!:
:sectanchors:

// Numbered sections
:sectnums:
:sectnumlevels: 2

// Links
:cc-by-nc-sa: http://creativecommons.org/licenses/by-nc-sa/4.0/

//************* END OF SETTINGS ******************
//************************************************


// Header
++++
<div align="center">
  <h1>BirdNET-Analyzer</h1>
  <p>Automated scientific audio data processing and bird ID.</p>
  <p><img src="https://tuc.cloud/index.php/s/xwKqoCmRDKzBCDZ/download/logo_box_birdnet.png" width="500px" /></p>
++++

// Badges
:license-badge: https://badgen.net/badge/License/CC-BY-NC-SA%204.0/green
:os-badge: https://badgen.net/badge/OS/Linux%2C%20Windows%2C%20macOS/blue
:species-badge: https://badgen.net/badge/Species/6512/blue
:twitter-badge: https://img.shields.io/twitter/follow/BirdNET_App
:reddit-badge: https://img.shields.io/reddit/subreddit-subscribers/BirdNET_Analyzer?style=social
// Mail icon from FontAwesome
:mail-badge: https://img.shields.io/badge/Mail us!-ccb--birdnet%40cornell.edu-yellow.svg?style=social&logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA1MTIgNTEyIj48IS0tISBGb250IEF3ZXNvbWUgUHJvIDYuNC4wIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlIChDb21tZXJjaWFsIExpY2Vuc2UpIENvcHlyaWdodCAyMDIzIEZvbnRpY29ucywgSW5jLiAtLT48cGF0aCBkPSJNNjQgMTEyYy04LjggMC0xNiA3LjItMTYgMTZ2MjIuMUwyMjAuNSAyOTEuN2MyMC43IDE3IDUwLjQgMTcgNzEuMSAwTDQ2NCAxNTAuMVYxMjhjMC04LjgtNy4yLTE2LTE2LTE2SDY0ek00OCAyMTIuMlYzODRjMCA4LjggNy4yIDE2IDE2IDE2SDQ0OGM4LjggMCAxNi03LjIgMTYtMTZWMjEyLjJMMzIyIDMyOC44Yy0zOC40IDMxLjUtOTMuNyAzMS41LTEzMiAwTDQ4IDIxMi4yek0wIDEyOEMwIDkyLjcgMjguNyA2NCA2NCA2NEg0NDhjMzUuMyAwIDY0IDI4LjcgNjQgNjRWMzg0YzAgMzUuMy0yOC43IDY0LTY0IDY0SDY0Yy0zNS4zIDAtNjQtMjguNy02NC02NFYxMjh6Ii8+PC9zdmc+

image:{license-badge}[CC BY-NC-SA 4.0, link={cc-by-nc-sa}]
image:{os-badge}[Supported OS, link=""]
image:{species-badge}[Number of species, link=""]

[.text-center]
image:{mail-badge}[Email, link=mailto:ccb-birdnet@cornell.edu, height=25]
image:https://img.shields.io/twitter/follow/BirdNET_App[Twitter Follow, link=https://twitter.com/BirdNET_App, height=25]
image:{reddit-badge}[Subreddit subscribers, link="https://reddit.com/r/BirdNET_Analyzer", height=25]

++++
</div>
++++

[discrete]
== Introduction

BirdNET is a deep learning solution for bird vocalization classification that can be used to monitor avian diversity.
This repo contains the BirdNET models and also scripts for processing large amounts of audio data or single audio files.
This is the most advanced version of BirdNET for acoustic analyses and we will keep this repository up-to-date with new models and improved interfaces to enable scientists with no computer science background to run the analysis.

Feel free to use BirdNET for your acoustic analyses and research.
If you do, please cite as:

----
@article{kahl2021birdnet,
  title={BirdNET: A deep learning solution for avian diversity monitoring},
  author={Kahl, Stefan and Wood, Connor M and Eibl, Maximilian and Klinck, Holger},
  journal={Ecological Informatics},
  volume={61},
  pages={101236},
  year={2021},
  publisher={Elsevier}
}
----

This work is licensed under a {cc-by-nc-sa}[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License].

[discrete]
== About

Developed by the https://www.birds.cornell.edu/ccb/[K. Lisa Yang Center for Conservation Bioacoustics] at the https://www.birds.cornell.edu/home[Cornell Lab of Ornithology] in collaboration with https://www.tu-chemnitz.de/index.html.en[Chemnitz University of Technology].

Go to https://birdnet.cornell.edu to learn more about the project.

Want to use BirdNET to analyze a large dataset? Don't hesitate to contact us: ccb-birdnet@cornell.edu

Follow us on Twitter https://twitter.com/BirdNET_App[@BirdNET_App]

We also have a discussion forum on https://reddit.com/r/BirdNET_Analyzer[Reddit] if you have a general question or just want to chat.

*Have a question, remark, or feature request? Please start a new issue thread to let us know. Feel free to submit a pull request.*


[discrete]
== Contents
toc::[]

== Model version update

[discrete]
==== V2.4, June 2023

* more than 6,000 species worldwide
* covers frequencies from 0 Hz to 15 kHz with two-channel spectrogram (one for low and one for high frequencies)
* 0.826 GFLOPs, 50.5 MB as FP32
* enhanced and optimized metadata model
* global selection of species (birds and non-birds) with 6,522 classes (incl. 10 non-event classes)

You can find a list of previous versions here: https://github.com/kahst/BirdNET-Analyzer/tree/main/checkpoints[BirdNET-Analyzer Model Version History]

== Showroom

BirdNET powers a number of fantastic community projects dedicated to bird song identification, all of which use models from this repository.
These are some highlights, make sure to check them out!

.Community projects
[cols="~,~", options="header"]
|===
| Project | Description

| image:https://tuc.cloud/index.php/s/cDqtQxo8yMRkNYP/download/logo_box_loggerhead.png[HaikuBox,300,link=https://haikubox.com]
| 
*HaikuBox* +
Once connected to your WiFi, Haikubox will listen for birds 24/7.
When BirdNET finds a match between its thousands of labeled sounds and the birdsong in your yard, it identifies the bird species and shares a three-second audio clip to the Haikubox website and smartphone app.

Learn more at: https://haikubox.com[HaikuBox.com]

| image:https://tuc.cloud/index.php/s/WKCZoE9WSjimDoe/download/logo_box_birdnet-pi.png[BirdNET-PI,300,link=https://birdnetpi.com]
| *BirdNET-Pi* +
Built on the TFLite version of BirdNET, this project uses pre-built TFLite binaries for Raspberry Pi to run on-device sound analyses.
It is able to recognize bird sounds from a USB sound card in realtime and share its data with the rest of the world.

Learn more at: https://birdnetpi.com[BirdNETPi.com]

| image:https://tuc.cloud/index.php/s/jDtyG9W36WwKpbR/download/logo_box_birdweather.png[BirdWeather,300,link=https://app.birdweather.com]
| *BirdWeather* +
This site was built to be a living library of bird vocalizations.
Using the BirdNET artificial neural network, BirdWeather is continuously listening to over 100 active stations around the world in real-time.

Learn more at: https://app.birdweather.com[BirdWeather.com]

| image:https://tuc.cloud/index.php/s/zpNkXJq7je3BKNE/download/logo_box_ecopi_bird.png[ecoPI:Bird,300,link=https://oekofor.netlify.app/en/portfolio/ecopi-bird_en/]
| *ecoPi:Bird* +
The ecoPi:Bird is a device for automated acoustic recordings of bird songs and calls, with a self-sufficient power supply.
It facilitates economical long-term monitoring, implemented with minimal personal requirements.

Learn more at: https://oekofor.netlify.app/en/portfolio/ecopi-bird_en/[oekofor.netlify.app]
|===

Working on a cool project that uses BirdNET? Let us know and we can feature your project here.

== Setup
=== Setup (Raven Pro)

If you want to analyze audio files without any additional coding or package install, you can now use https://ravensoundsoftware.com/software/raven-pro/[Raven Pro software] to run BirdNET models.
After download, BirdNET is available through the new "Learning detector" feature in Raven Pro.
For more information on how to use this feature, please visit the https://ravensoundsoftware.com/article-categories/learning-detector/[Raven Pro Knowledge Base].

=== Setup (birdnetlib)

The easiest way to setup BirdNET on your machine is to install https://pypi.org/project/birdnetlib/[birdnetlib] through pip.
However, this requires running Linux operating system as the implementation uses tflite-runtime.

Under Linux:

[source,sh]
----
sudo apt install --yes ffmpeg
----

Setup a virtual environment:

[source,sh]
----
python3.9 -m venv venv
venv/bin/python -m pip install birdnetlib
----

Upgrade Python packaging tools:

[source,sh]
----
venv/bin/python -m pip install --upgrade setuptools pip wheel
----

Install required Python packages:

[source,sh]
----
venv/bin/python -m pip install birdnetlib librosa resampy TensorFlow
----

Create a birdnetlib application:

[source,python]
----
# main.py

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime


def main():
    # Load and initialize the BirdNET-Analyzer models
    analyzer = Analyzer()

    # Create a recoding object from a file
    recording = Recording(
        analyzer,
        "example/soundscape.wav",
        lat=35.4244,
        lon=-120.7463,
        date=datetime(year=2022, month=5, day=10), # use date or week_48
        min_conf=0.25,
    )

    # Analyze the recording object
    recording.analyze()

    # Print the detections
    print(recording.detections)


if __name__ == '__main__':
    main()
----

Run your application:

[source,sh]
----
venv/bin/python main.py
----

The output should look something like this:

----
load model
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
Model loaded.
Labels loaded.
Labels loaded.
load_species_list_model
Meta model loaded.
read_audio_data
read_audio_data: complete, read  40 chunks.
analyze_recording example/soundscape.wav
recording has lon/lat
set_predicted_species_list_from_position
return_predicted_species_list
18
130 species loaded.
[{'common_name': 'House Finch', 'scientific_name': 'Haemorhous mexicanus', 'start_time': 9.0, 'end_time': 12.0, 'confidence': 0.5169966220855713}, {'common_name': 'Dark-eyed Junco', 'scientific_name': 'Junco hyemalis', 'start_time': 27.0, 'end_time': 30.0, 'confidence': 0.3839336335659027}, {'common_name': 'Dark-eyed Junco', 'scientific_name': 'Junco hyemalis', 'start_time': 33.0, 'end_time': 36.0, 'confidence': 0.2508070766925812}, {'common_name': 'Dark-eyed Junco', 'scientific_name': 'Junco hyemalis', 'start_time': 36.0, 'end_time': 39.0, 'confidence': 0.397082656621933}, {'common_name': 'Dark-eyed Junco', 'scientific_name': 'Junco hyemalis', 'start_time': 42.0, 'end_time': 45.0, 'confidence': 0.7388747930526733}, {'common_name': 'House Finch', 'scientific_name': 'Haemorhous mexicanus', 'start_time': 72.0, 'end_time': 75.0, 'confidence': 0.3488200902938843}, {'common_name': 'House Finch', 'scientific_name': 'Haemorhous mexicanus', 'start_time': 84.0, 'end_time': 87.0, 'confidence': 0.45684170722961426}]
----

For more examples and documentation, make sure to visit https://pypi.org/project/birdnetlib/[pypi.org/project/birdnetlib/].
For any feature request or questions regarding *birdnetlib*, please contact link:mailto:joe.weiss@gmail.com[Joe Weiss] or add an issue or PR at https://github.com/joeweiss/birdnetlib[github.com/joeweiss/birdnetlib].

=== Setup (Ubuntu)

The only LTS version of Ubuntu that can be used with BirdNET out of the box is 20.04:

- Ubuntu 18.04 preinstalls Python 3.6, which is incompatible with BirdNET.
- Ubuntu 20.04 preinstalls Python 3.8, which is compatible with BirdNET.
- Ubuntu 22.04 preinstalls Python 3.10, which is incompatible with BirdNET.

If your Ubuntu version preinstalls a Python version that is incompatible with BirdNET, install Python 3.9 from the deadsnakes personal package archive:

[source,sh]
----
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install --yes python3.9 python3.9-dev python3.9-venv
----

If your Ubuntu version preinstalls a Python version that is compatible with BirdNET, install the :

[source,sh]
----
sudo apt-get update
sudo apt-get install --yes python3.10 python3.10-dev python3.10-venv
----

NOTE: The following assumes that you are using Python 3.9, so adapt the commands if you need.

Install Ubuntu packages:

[source,sh]
----
sudo apt install --yes ffmpeg
----

Clone the repository:

[source,sh]
----
git clone https://github.com/kahst/BirdNET-Analyzer.git
cd BirdNET-Analyzer
----

Setup a virtual environment:

[source,sh]
----
python3.9 -m venv venv
----

Upgrade Python packaging tools:

[source,sh]
----
venv/bin/python -m pip install --upgrade pip setuptools wheel
----

Install Python packages. You can opt to either install the TFLite runtime (recommended) or TensorFlow:

[source,sh]
----
venv/bin/python -m pip install -r requirements.d/test_lite.in
----

or

[source,sh]
----
venv/bin/python -m pip install -r requirements.d/test.in
----

Freeze requirements:

[source,sh]
----
venv/bin/python -m pip freeze --all > requirements.d/test.txt
----

[source,sh]
----
wget https://github.com/Bengt/BirdNET-Training-Data/archive/refs/heads/main.zip
unzip main.zip
rm -rf main.zip
----

=== Setup (Windows)

Before you attempt to setup BirdNET-Analyzer on your Windows machine, please consider downloading our fully-packaged version that does not require you to install any additional packages and can be run "as-is".

You can download this version here: https://tuc.cloud/index.php/s/myHcJNsDsMrDqMM/download/BirdNET-Analyzer.zip[BirdNET-Analyzer Windows]

. Download the zip file
. Before unpacking, make sure to right-click the zip-file, select "Properties" and check the box "Unblock" under "Security" at the bottom of the "General" tab.
 ** If Windows does not display this option, the file can be unblocked with the PowerShell 7 command `Unblock-File -Path .\BirdNET-Analyzer.zip`
. Unpack the zip-file
. Navigate to the extracted folder named "BirdNET-Analyzer"
. You can start the analysis through the command prompt with `+BirdNET-Analyzer.exe --i "path\to\folder" ...+` (see <<usage-cli,Usage (CLI) section>> for more details), or you can launch `BirdNET-Analyzer-GUI.exe` to start the analysis through a basic GUI.

For more advanced use cases (e.g., hosting your own API server), follow these steps to set up BirdNET-Analyzer on your Windows machine:

Install Python 3.9 or higher (has to be 64bit version)

* Download and run installer: https://www.python.org/downloads/release/python-390/[Download Python installer]

WARNING: :exclamation:**Make sure to check: &#x2611; "Add path to environment variables" during install**:exclamation:

Install TensorFlow (has to be 2.5 or later), Librosa and NumPy

* Open command prompt with *`Win + S`* type "command" and click on "Command Prompt"
* Type `pip install --upgrade pip`
* Type `pip install librosa resampy numpy==1.20`
* Install TensorFlow by typing `pip install TensorFlow`

NOTE: You might need to run the command prompt as "administrator".
Type *`Win + S`*, search for command prompt and then right-click, select "Run as administrator".

Install Visual Studio Code (optional)

* Download and install VS Code: https://code.visualstudio.com/sha/download?build=stable&os=win32-x64-user[Download VS Code installer]
* Select all available options during install

Install BirdNET using Git (for simple download see below)

* Download and install Git Bash: https://github.com/git-for-windows/git/releases/download/v2.34.1.windows.1/Git-2.34.1-64-bit.exe[Download Git Bash installer]
* Select Visual Studio Code as default editor (optional)
* Keep all other settings as recommended
* Create folder in personal directory called "Code" (or similar)
* Change to folder and right click, launch "Git bash here"
* Type `+git clone https://github.com/kahst/BirdNET-Analyzer.git+`
* Keep BirdNET updated by running `git pull` for BirdNET-Analyzer folder occasionally

Install BirdNET from zip

* Download BirdNET: https://github.com/kahst/BirdNET-Analyzer/archive/refs/heads/main.zip[Download BirdNET Zip-file]
* Unpack zip file (e.g., in personal directory)
* Keep BirdNET updated by re-downloading the zip file occasionally and overwrite existing files

Run BirdNET from command line

* Open command prompt with *`Win + S`* type "command" and click on "Command Prompt"
* Navigate to the folder where you installed BirdNET (cd path\to\BirdNET-Analyzer)
* See <<usage-cli,Usage (CLI) section>> for command line arguments

NOTE: With Visual Studio Code installed, you can right-click the BirdNET-Analyzer folder and select "Open with Code".
With proper extensions installed (View -> Extensions -> Python) you will be able to run all scripts from within VS Code.

=== Setup (macOS)

NOTE: Installation was only tested on a M1 chip.
Feedback on older Intel CPUs or newer M2 chips is welcome!

==== Requirements

Xcode command-line tools:

[source,sh]
----
xcode-select --install
----

Conda (Apple silicon):

[source,sh]
----
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/Downloads/Miniconda3-latest-MacOSX-arm64.sh
bash ~/Downloads/Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/miniconda
----

Conda (x86_64):

[source,sh]
----
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o ~/Downloads/Miniconda3-latest-MacOSX-x86_64.sh
bash ~/Downloads/Miniconda3-latest-MacOSX-x86_64.sh -b -p $HOME/miniconda
----

The installer prompts "`Do you wish the installer to initialize Miniconda3 by running conda init?`" We recommend "`yes`".

Add the `conda-forge` channel:

[source,sh]
----
conda config --add channels conda-forge
----

==== Create Conda Environment

[source,sh]
----
conda create -n birdnet-analyzer python=3.10 -c conda-forge -y
conda activate birdnet-analyzer
----

==== Install dependencies

Apple silicon only:

[source,sh]
----
conda install -c apple TensorFlow-deps
----

TensorFlow for macOS and Metal plug-in:

[source,sh]
----
python -m pip install TensorFlow-macos TensorFlow-metal
----

Librosa and ffmpeg:

[source,sh]
----
conda install -c conda-forge librosa resampy -y
----

==== Verify

Clone the git repository if you have not done that yet:

[source,sh]
----
git clone https://github.com/kahst/BirdNET-Analyzer.git
cd BirdNET-Analyzer
----

Run the example.
It will take a while the first time you run it.
Subsequent runs will be faster.

[source,sh]
----
PYTHONPATH=. python birdnet/analysis/main.py
----

NOTE: Now, you can install and use <<setup-birdnetlib,birdnetlib>>.

== Building

BirdNet-Analyzer can be packaged into three portable applications using
PyInstaller. These are:

- "BirdNET-Analyzer-Analyse",
- "BirdNET-Analyzer-GUI", and
- "BirdNET-Analyzer-Full"

> NOTE: PyInstaller builds executables in the context it is run in.
> So, compilation across operating systems or Python versions is not supported.

=== Building BirdNET-Analyzer-Analyse

Under Linux:

```bash
venv/bin/python pyinstaller_analyze.py
```

Under macOS:

```bash
venv/bin/python pyinstaller_analyze.py
```

Under Windows:

```bash
venv\Scripts\python.exe pyinstaller_analyze.py
```

=== Building BirdNET-Analyzer-GUI

Under Linux:

```bash
venv/bin/python pyinstaller_gui.py
```

Under macOS:

```bash
venv/bin/python pyinstaller_gui.py
```

Under Windows:

```bash
venv\Scripts\python.exe pyinstaller_gui.py
```

=== Building BirdNET-Analyzer-Full

Under Linux:

```bash
venv/bin/python pyinstaller_full.py
```

Under macOS:

```bash
venv/bin/python pyinstaller_full.py
```

Under Windows:

```bash
venv\Scripts\python.exe pyinstaller_full.py
```

== Executing the Portables

The executables can be executed in the current directory:

=== Executing the Analysis Portable

Under Linux:

```bash
build/BirdNET-Analyzer-Analysis/BirdNET-Analyzer-Analysis --help
```

Under macOS:

```bash
build/BirdNET-Analyzer-Analysis/BirdNET-Analyzer-Analysis --help
```

Under Windows:

```bash
build\BirdNET-Analyzer-Analysis\BirdNET-Analyzer-Analysis.exe --help
```

=== Executing the GUI Portable

Under Linux:

```bash
build/BirdNET-Analyzer-GUI/BirdNET-Analyzer-GUI --help
```

Under macOS:

```bash
build/BirdNET-Analyzer-GUI/BirdNET-Analyzer-GUI --help
```

Under Windows:

```bash
build\BirdNET-Analyzer-GUI\BirdNET-Analyzer-GUI.exe --help
```

== Usage

=== Usage (CLI)

. Inspect config file for options and settings, especially inference settings.
Specify a custom species list if needed and adjust the number of threads TFLite can use to run the inference.
. Run `analyzer.py` to analyze an audio file.
You need to set paths for the audio file and selection table output.
Here is an example:
+
[source,sh]
----
PYTHONPATH=. python3 birdnet/analysis/main.py --i /path/to/audio/folder --o /path/to/output/folder
----
+
NOTE: Your custom species list has to be named 'species_list.txt' and the folder containing the list needs to be specified with `--slist /path/to/folder`.
You can also specify the number of CPU threads that should be used for the analysis with `--threads <Integer>` (e.g., `--threads 16`).
If you provide GPS coordinates with `--lat` and `--lon`, the custom species list argument will be ignored.
+

Here's a complete list of all command line arguments:
+
----
--i, Path to input file or folder. If this is a file, --o needs to be a file too.
--o, Path to output file or folder. If this is a file, --i needs to be a file too.
--lat, Recording location latitude. Set -1 to ignore.
--lon, Recording location longitude. Set -1 to ignore.
--week, Week of the year when the recording was made. Values in [1, 48] (4 weeks per month). Set -1 for year-round species list.
--slist, Path to species list file or folder. If folder is provided, species list needs to be named "species_list.txt". If lat and lon are provided, this list will be ignored.
--sensitivity, Detection sensitivity; Higher values result in higher sensitivity. Values in [0.5, 1.5]. Defaults to 1.0.
--min_conf, Minimum confidence threshold. Values in [0.01, 0.99]. Defaults to 0.1.
--overlap, Overlap of prediction segments. Values in [0.0, 2.9]. Defaults to 0.0.
--rtype, Specifies output format. Values in ['table', 'audacity', 'r', 'kaleidoscope', 'csv']. Defaults to 'table' (Raven selection table).
--threads, Number of CPU threads.
--batchsize, Number of samples to process at the same time. Defaults to 1.
--locale, Locale for translated species common names. Values in ['af', 'de', 'it', ...] Defaults to 'en'.
--sf_thresh, Minimum species occurrence frequency threshold for location filter. Values in [0.01, 0.99]. Defaults to 0.03.
--classifier, Path to custom trained classifier. Defaults to None. If set, --lat, --lon and --locale are ignored.
----
+
Here are two example commands to run this BirdNET version:
+
[source,sh]
----
PYTHONPATH=. python3 birdnet/analysis/main.py --i example/ --o example/ --slist example/ --min_conf 0.5 --threads 4

PYTHONPATH=. python3 birdnet/analysis/main.py --i example/ --o example/ --lat 42.5 --lon -76.45 --week 4 --sensitivity 1.0
----
+
. Run `embeddings.py` to extract feature embeddings instead of class predictions.
Result file will contain timestamps and lists of float values representing the embedding for a particular 3-second segment.
Embeddings can be used for clustering or similarity analysis.
Here is an example:
+
[,sh]
----
python3 embeddings.py --i example/ --o example/ --threads 4 --batchsize 16
----
+
Here's a complete list of all command line arguments:
+
----
--i, Path to input file or folder. If this is a file, --o needs to be a file too.
--o, Path to output file or folder. If this is a file, --i needs to be a file too.
--overlap, Overlap of prediction segments. Values in [0.0, 2.9]. Defaults to 0.0.
--threads, Number of CPU threads.
--batchsize, Number of samples to process at the same time. Defaults to 1.
----
+
. After the analysis, run `segments.py` to extract short audio segments for species detections to verify results.
This way, it might be easier to review results instead of loading hundreds of result files manually.
+
Here's a complete list of all command line arguments:
+
----
--audio, Path to folder containing audio files.
--results, Path to folder containing result files.
--o, Output folder path for extracted segments.
--min_conf, Minimum confidence threshold. Values in [0.01, 0.99]. Defaults to 0.1.
--max_segments, Number of randomly extracted segments per species.
--seg_length, Length of extracted segments in seconds. Defaults to 3.0.
--threads, Number of CPU threads.
----
+
. When editing your own `species_list.txt` file, make sure to copy species names from the labels file of each model.
+
You can find label files in the checkpoints folder, e.g., `checkpoints/V2.3/BirdNET_GLOBAL_3K_V2.3_Labels.txt`.
+
Species names need to consist of `scientific name_common name` to be valid.
+
. You can generate a species list for a given location using `species.py` in case you need it for reference.
Here is an example:
+
[,sh]
----
python3 species.py --o example/species_list.txt --lat 42.5 --lon -76.45 --week 4
----
+
Here's a complete list of all command line arguments:
+
----
--o, Path to output file or folder. If this is a folder, file will be named 'species_list.txt'.
--lat, Recording location latitude.
--lon, Recording location longitude.
--week, Week of the year when the recording was made. Values in [1, 48] (4 weeks per month). Set -1 for year-round species list.
--threshold, Occurrence frequency threshold. Defaults to 0.05.
--sortby, Sort species by occurrence frequency or alphabetically. Values in ['freq', 'alpha']. Defaults to 'freq'.
----
+
. This is a very basic version of the analysis workflow, you might need to adjust it to your own needs.
. Please open an issue to ask for new features or to document unexpected behavior.
. I will keep models up to date and upload new checkpoints whenever there is an improvement in performance.
I will also provide quantized and pruned model files for distribution.

=== Usage (Docker)

Install docker for Ubuntu:

[source,sh]
----
sudo apt install docker.io
----

Build Docker container:

[source,sh]
----
sudo docker build -t birdnet .
----

NOTE: You need to run docker build again whenever you make changes to the script.

In order to pass a directory that contains your audio files to the docker file, you need to mount it inside the docker container with `-v /my/path:/mount/path` before you can run the container.

You can run the container for the provided example soundscapes with:

[source,sh]
----
sudo docker run -v $PWD/example:/audio birdnet birdnet/analysis/main.py --i audio --o audio --slist audio
----

You can adjust the directory that contains your recordings by providing an absolute path:

[source,sh]
----
sudo docker run -v /path/to/your/audio/files:/audio birdnet birdnet/analysis/main.py --i audio --o audio --slist audio
----

You can also mount more than one drive, e.g., if input and output folder should be different:

[source,sh]
----
sudo docker run -v /path/to/your/audio/files:/input -v /path/to/your/output/folder:/output birdnet birdnet/analysis/main.py --i input --o output --slist input
----

See <<usage-cli,Usage (CLI) section>> above for more command line arguments, all of them will work with Docker version.

NOTE: If you like to specify a species list (which will be used as post-filter and needs to be named 'species_list.txt'), you need to put it into a folder that also has to be mounted.

=== Usage (Server)

You can host your own analysis service and API by launching the `server.py` script.
This will allow you to send files to this server, store submitted files, analyze them and send detection results back to a client.
This could be a local service, running on a desktop PC, or a remote server.
The API can be accessed locally or remotely through a browser or Python client (or any other client implementation).

. Install one additional package with `pip3 install bottle`.
. Start the server with `python3 server.py`.
You can also specify a host name or IP and port number, e.g., `python3 server.py --host localhost --port 8080`.
+
Here's a complete list of all command line arguments:
+
----
--host, Host name or IP address of API endpoint server. Defaults to '0.0.0.0'.
--port, Port of API endpoint server. Defaults to 8080.
--spath, Path to folder where uploaded files should be stored. Defaults to '/uploads'.
--threads, Number of CPU threads for analysis. Defaults to 4.
--locale, Locale for translated species common names. Values in ['af', 'de', 'it', ...] Defaults to 'en'.
----
+
NOTE: The server is single-threaded, so you'll need to start multiple instances for higher throughput.
This service is intented for short audio files (e.g., 1-10 seconds).
+
. Query the API with a client.
You can use the provided Python client or any other client implementation.
Request payload needs to be `multipart/form-data` with the following fields: `audio` for raw audio data as byte code, and `meta` for additional information on the audio file.
Take a look at our example client implementation in the `client.py` script.
+
This script will read an audio file, generate metadata from command line arguments and send it to the server.
The server will then analyze the audio file and send back the detection results which will be stored as a JSON file.
+
Here's a complete list of all command line arguments:
+
----
--host, Host name or IP address of API endpoint server.
--port, Port of API endpoint server.
--i, Path to file that should be analyzed.
--o, Path to result file. Leave blank to store with audio file.
--lat, Recording location latitude. Set -1 to ignore.
--lon, Recording location longitude. Set -1 to ignore.
--week, Week of the year when the recording was made. Values in [1, 48] (4 weeks per month). Set -1 for year-round species list.
--overlap, Overlap of prediction segments. Values in [0.0, 2.9]. Defaults to 0.0.
--sensitivity, Detection sensitivity; Higher values result in higher sensitivity. Values in [0.5, 1.5]. Defaults to 1.0.
--pmode, Score pooling mode. Values in ['avg', 'max']. Defaults to 'avg'.
--num_results, Number of results per request.
--sf_thresh, Minimum species occurrence frequency threshold for location filter. Values in [0.01, 0.99]. Defaults to 0.03.
--save, Define if files should be stored on server. Values in [True, False]. Defaults to False.
----
+
. Parse results from the server.
The server will send back a JSON response with the detection results.
The response also contains a `msg` field, indicating `success` or `error`.
Results consist of a sorted list of (species, score) tuples.
+
This is an example response:
+

[source,json]
----
{"msg": "success", "results": [["Poecile atricapillus_Black-capped Chickadee", 0.7889], ["Spinus tristis_American Goldfinch", 0.5028], ["Junco hyemalis_Dark-eyed Junco", 0.4943], ["Baeolophus bicolor_Tufted Titmouse", 0.4345], ["Haemorhous mexicanus_House Finch", 0.2301]]}
----
+
NOTE: Let us know if you have any questions, suggestions, or feature requests.
Also let us know when hosting an analysis service - we would love to give it a try.

=== Usage (GUI)

We provide a very basic GUI which lets you launch the analysis through a web interface.

.Web based GUI
image::https://tuc.cloud/index.php/s/QyBczrWXCrMoaRC/download/analyzer_gui.png[GUI screenshot]

. You need to install two additional packages in order to use the GUI with `pip3 install pywebview gradio`
. Launch the GUI with `python3 gui.py`.
. Set all folders and parameters, after that, click 'Analyze'.

== Training

You can train your own custom classifier on top of BirdNET.
This is useful if you want to detect species that are not included in the default species list.
You can also use this to train a classifier for a specific location or season.
All you need is a dataset of labeled audio files, organized in folders by species (we use folder names as labels).
*This also works for non-bird species, as long as you have a dataset of labeled audio files*.
Audio files will be resampled to 48 kHz and converted into 3-second segments (we will use the center 3-second segment if the file is longer, we will pad with random noise if the file is shorter).
We recommend using at least 100 audio files per species (although training also works with less data).
You can download a sample training data set https://drive.google.com/file/d/16hgka5aJ4U69ane9RQn_quVmgjVY2AY5[here].

. Collect training data and organize in folders based on species names.
. Species labels should be in the format `<scientific name>_<species common name>` (e.g., `Poecile atricapillus_Black-capped Chickadee`), but other formats work as well.
. It can be helpful to include a non-event class.
If you name a folder 'Noise', 'Background', 'Other' or 'Silence', it will be treated as a non-event class.
. Run the training script with `python3 bridnet/training/main.py --i <path to training data folder> --o <path to trained classifier model output>`.
+
Here is a list of all command line arguments:
+
----
--i, Path to training data folder. Subfolder names are used as labels.
--o, Path to trained classifier model output.
--epochs, Number of training epochs. Defaults to 100.
--batch_size, Batch size. Defaults to 32.
--learning_rate, Learning rate. Defaults to 0.01.
--hidden_units, Number of hidden units. Defaults to 0. If set to >0, a two-layer classifier is used.
----
+
. After training, you can use the custom trained classifier with the `--classifier` argument of the `birdnet/analysis/main.py` script.
+
NOTE: Adjusting hyperparameters (e.g., number of hidden units, learning rate, etc.) can have a big impact on the performance of the classifier.
We recommend trying different hyperparameter settings.
+
Example usage (when downloading and unzipping the sample training data set):
+

[source,sh]
----
python3 birdnet/training/main.py --i train_data/ --o checkpoints/custom/Custom_Classifier.tflite
python3 birdnet/analysis/main.py --classifier checkpoints/custom/Custom_Classifier.tflite
----
+
NOTE: Setting a custom classifier will also set the new labels file.
Due to these custom labels, the location filter and locale will be disabled.

== Funding

This project is supported by Jake Holshuh (Cornell class of `'69) and The Arthur Vining Davis Foundations.
Our work in the K.
Lisa Yang Center for Conservation Bioacoustics is made possible by the generosity of K.
Lisa Yang to advance innovative conservation technologies to inspire and inform the conservation of wildlife and habitats. 

The German Federal Ministry of Education and Research is funding the development of BirdNET through the project "BirdNET+" (FKZ 01|S22072).
Additionally, the German Federal Ministry of Environment, Nature Conservation and Nuclear Safety is funding the development of BirdNET through the project "DeepBirdDetect" (FKZ 67KI31040E).

== Partners

BirdNET is a joint effort of partners from academia and industry.
Without these partnerships, this project would not have been possible.
Thank you!

.Our partners
image::https://tuc.cloud/index.php/s/KSdWfX5CnSRpRgQ/download/box_logos.png[Logos of all partners]
