<h1 align="center">BirdNET-Analyzer</h1>
<p align="center">Automated scientific audio data processing and bird ID.</p>
<p align="center"><img src="https://tuc.cloud/index.php/s/xwKqoCmRDKzBCDZ/download/logo_box_birdnet.png" width="500px" /></p>

[![CC BY-NC-SA 4.0][license-badge]][cc-by-nc-sa] 
![Supported OS][os-badge]
![Number of species][species-badge]

[license-badge]: https://badgen.net/badge/License/CC-BY-NC-SA%204.0/green
[os-badge]: https://badgen.net/badge/OS/Linux%2C%20Windows/blue
[species-badge]: https://badgen.net/badge/Species/3327/blue

## Introduction
This repo contains BirdNET models and scripts for processing large amounts of audio data or single audio files. This is the most advanced version of BirdNET for acoustic analyses and we will keep this repository up-to-date with new models and improved interfaces to enable scientists with no CS background to run the analysis.

Feel free to use BirdNET for your acoustic analyses and research. If you do, please cite as:

```
@article{kahl2021birdnet,
  title={BirdNET: A deep learning solution for avian diversity monitoring},
  author={Kahl, Stefan and Wood, Connor M and Eibl, Maximilian and Klinck, Holger},
  journal={Ecological Informatics},
  volume={61},
  pages={101236},
  year={2021},
  publisher={Elsevier}
}
```

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/

## About

Developed by the [K. Lisa Yang Center for Conservation Bioacoustics](https://www.birds.cornell.edu/ccb/) at the [Cornell Lab of Ornithology](https://www.birds.cornell.edu/home).

Go to https://birdnet.cornell.edu to learn more about the project.

Follow us on Twitter [@BirdNET_App](https://twitter.com/BirdNET_App).

Want to use BirdNET to analyze a large dataset? Don't hesitate to contact us: ccb-birdnet@cornell.edu

<b>Have a question, remark or feature request? Please start a new issue thread to let us know. Feel free to submit pull request.</b>

We also have a discussion forum on Reddit if you have a general question or just want to chat. Check it out here: [reddit.com/r/BirdNET_Analyzer](https://www.reddit.com/r/BirdNET_Analyzer)

## Contents

[Model version update](#model-version-update)  
[Showroom](#showroom)    
[Setup (Ubuntu)](#setup-ubuntu)  
[Setup (Windows)](#setup-windows)  
[Usage](#usage)  
[Usage (Docker)](#usage-docker)  
[Usage (Server)](#usage-server)   
[Usage (GUI)](#usage-gui)  
[Funding](#funding)  
[Partners](#partners)

## Model version update

**V2.2, Aug 2022**

- smaller (21.3 MB vs. 29.5 MB as FP32) and faster (1.31 vs 2.03 GFLOPs) than V2.1
- maintains same accuracy as V2.1 despite more classes
- global selection of species (birds and non-birds) with 3,337 classes (incl. 10 non-event classes)

You can find a list of previous versions here: [BirdNET-Analyzer Model Version History](https://github.com/kahst/BirdNET-Analyzer/tree/main/checkpoints)

## Showroom

BirdNET powers a number of fantastic community projects dedicated to bird song identification, all of which use models from this repository. These are some highlights, make sure to check them out!

| Project | Description |
| :--- | :--- |
| <a href="https://haikubox.com"><img src="https://tuc.cloud/index.php/s/cDqtQxo8yMRkNYP/download/logo_box_loggerhead.png" /></a> | <b>HaikuBox</b><p>Once connected to your WiFi, Haikubox will listen for birds 24/7.  When BirdNET finds a match between its thousands of labeled sounds and the birdsong in your yard, it identifies the bird species and shares a three-second audio clip to the Haikubox website and smartphone app.</p> Learn more at: [HaikuBox.com](https://haikubox.com)|
| <a href="https://birdnetpi.com"><img src="https://tuc.cloud/index.php/s/WKCZoE9WSjimDoe/download/logo_box_birdnet-pi.png" /></a> | <b>BirdNET-Pi</b><p>Built on the TFLite version of BirdNET, this project uses pre-built TFLite binaries for Raspberry Pi to run on-device sound analyses. It is able to recognize bird sounds from a USB sound card in realtime and share its data with the rest of the world.</p> Learn more at: [BirdNETPi.com](https://birdnetpi.com)|
| <a href="https://app.birdweather.com"><img src="https://tuc.cloud/index.php/s/jDtyG9W36WwKpbR/download/logo_box_birdweather.png" /></a> | <b>BirdWeather</b><p>This site was built to be a living library of bird vocalizations. Using the BirdNET artificial neural network, BirdWeather is continuously listening to over 100 active stations around the world in real-time.</p> Learn more at: [BirdWeather.com](https://app.birdweather.com)|
| <a href="https://oekofor.netlify.app/en/portfolio/ecopi-bird_en/"><img src="https://tuc.cloud/index.php/s/zpNkXJq7je3BKNE/download/logo_box_ecopi_bird.png" /></a> | <b>ecoPi:Bird</b><p>The ecoPi:Bird is a device for automated acoustic recordings of bird songs and calls, with a self-sufficient power supply. It facilitates economical long-term monitoring, implemented with minimal personal requirements.</p> Learn more at: [oekofor.netlify.app](https://oekofor.netlify.app/en/portfolio/ecopi-bird_en/)|

Working on a cool project that uses BirdNET? Let us know and we can feature your project here.

## Setup (Ubuntu)

Install Python 3:
```
sudo apt-get update
sudo apt-get install python3-dev python3-pip
sudo pip3 install --upgrade pip
```

Install TFLite runtime (recommended) or Tensorflow (has to be 2.5 or later):
```
sudo pip3 install tflite-runtime

OR

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
## Setup (Windows)

Before you attempt to setup BirdNET-Analyzer on your Windows machine, please consider downloading our fully-packaged version that does not require you to install any additional packages and can be run "as-is".

You can download this version here: [BirdNET-Analyzer Windows](https://tuc.cloud/index.php/s/myHcJNsDsMrDqMM/download/BirdNET-Analyzer.zip)

1. Download the zip file
2. Before unpacking, make sure to right-click the zip-file, select "Properties" and check the box "Unblock" under "Security" at the bottom of the "General" tab.
3. Unpack the zip-file
4. Navigate to the extracted folder named "BirdNET-Analyzer"
5. You can start the analysis through the command prompt with `BirdNET-Analyzer.exe --i "path\to\folder" ...` (see [Usage section](#usage) for more details), or you can launch `BirdNET-Analyzer-GUI.exe` to start the analysis through a basic GUI.

<b>NOTE</b>: You can edit the provided `run.bat` file based on your requirements (right-click --> "Edit") and then simply double-click this file to start the analysis.

For more advanced use cases (e.g., hosting your own API server), follow these steps to set up BirdNET-Analyzer on your Windows machine:

Install Python 3.8 (has to be 64bit version)

- Download and run installer: [Download Python installer](https://www.python.org/ftp/python/3.8.0/python-3.8.0-amd64.exe)
- :exclamation:<b>Make sure to check "Add path to environment variables" during install</b>:exclamation:

Install Tensorflow (has to be 2.5 or later), Librosa and NumPy

- Open command prompt with "Win+S" type "command" and click on "Command Prompt"
- Type `pip install --upgrade pip`
- Type `pip install librosa numpy==1.20`
- Install Tensorflow by typing `pip install tensorflow`

<b>NOTE</b>: You might need to run the command prompt as "administrator". Type "Win+S", search for command prompt and then right-click, select "Run as administrator".

Install Visual Studio Code (optional)

- Download and install VS Code: [Download VS Code installer](https://code.visualstudio.com/sha/download?build=stable&os=win32-x64-user)
- Select all available options during install

Install BirdNET using Git (for simple download see below)

- Download and install Git Bash: [Download Git Bash installer](https://github.com/git-for-windows/git/releases/download/v2.34.1.windows.1/Git-2.34.1-64-bit.exe)
- Select Visual Studio Code as default editor (optional)
- Keep all other settings as recommended
- Create folder in personal directory called "Code" (or similar)
- Change to folder and right click, launch "Git bash here"
- Type `git clone https://github.com/kahst/BirdNET-Analyzer.git`
- Keep BirdNET updated by running `git pull` for BirdNET-Analyzer folder occasionally

Install BirdNET from zip

- Download BirdNET: [Download BirdNET Zip-file](https://github.com/kahst/BirdNET-Analyzer/archive/refs/heads/main.zip)
- Unpack zip file (e.g., in personal directory)
- Keep BirdNET updated by re-downloading the zip file occasionally and overwrite existing files

Run BirdNET from command line

- Open command prompt with "Win+S" type "command" and click on "Command Prompt"
- Navigate to the folder where you installed BirdNET (cd path\to\BirdNET-Analyzer)
- See "Usage" section for command line arguments

<b>NOTE</b>: With Visual Studio Code installed, you can right-click the BirdNET-Analyzer folder and select "Open with Code". With proper extensions installed (View --> Extensions --> Python) you will be able to run all scripts from within VS Code.

## Usage

1. Inspect config file for options and settings, especially inference settings. Specify a custom species list if needed and adjust the number of threads TFLite can use to run the inference.

2. Run `analyzer.py` to analyze an audio file. You need to set paths for the audio file and selection table output. Here is an example:

```
python3 analyze.py --i /path/to/audio/folder --o /path/to/output/folder
```

<b>NOTE</b>: Your custom species list has to be named 'species_list.txt' and the folder containing the list needs to be specified with `--slist /path/to/folder`. You can also specify the number of CPU threads that should be used for the analysis with `--threads <Integer>` (e.g., `--threads 16`). If you provide GPS coordinates with `--lat` and `--lon`, the custom species list argument will be ignored.

Here's a complete list of all command line arguments:

```
--i, Path to input file or folder. If this is a file, --o needs to be a file too.
--o, Path to output file or folder. If this is a file, --i needs to be a file too.
--lat, Recording location latitude. Set -1 to ignore.
--lon, Recording location longitude. Set -1 to ignore.
--week, Week of the year when the recording was made. Values in [1, 48] (4 weeks per month). Set -1 for year-round species list.
--slist, Path to species list file or folder. If folder is provided, species list needs to be named "species_list.txt". If lat and lon are provided, this list will be ignored.
--sensitivity, Detection sensitivity; Higher values result in higher sensitivity. Values in [0.5, 1.5]. Defaults to 1.0.
--min_conf, Minimum confidence threshold. Values in [0.01, 0.99]. Defaults to 0.1.
--overlap, Overlap of prediction segments. Values in [0.0, 2.9]. Defaults to 0.0.
--rtype, Specifies output format. Values in ['table', 'audacity', 'r', 'csv']. Defaults to 'table' (Raven selection table).
--threads, Number of CPU threads.
--batchsize, Number of samples to process at the same time. Defaults to 1.
--locale, Locale for translated species common names. Values in ['af', 'de', 'it', ...] Defaults to 'en'.
--sf_thresh, Minimum species occurrence frequency threshold for location filter. Values in [0.01, 0.99]. Defaults to 0.03.
```

Here are two example commands to run this BirdNET version:

```
python3 analyze.py --i example/ --o example/ --slist example/ --min_conf 0.5 --threads 4

python3 analyze.py --i example/ --o example/ --lat 42.5 --lon -76.45 --week 4 --sensitivity 1.0
```

3. Run `embeddings.py` to extract feature embeddings instead of class predictions. Result file will contain timestamps and lists of float values representing the embedding for a particular 3-second segment. Embeddings can be used for clustering or similarity analysis. Here is an example:

```
python3 embeddings.py --i example/ --o example/ --threads 4 --batchsize 16
```

Here's a complete list of all command line arguments:

```
--i, Path to input file or folder. If this is a file, --o needs to be a file too.
--o, Path to output file or folder. If this is a file, --i needs to be a file too.
--overlap, Overlap of prediction segments. Values in [0.0, 2.9]. Defaults to 0.0.
--threads, Number of CPU threads.
--batchsize, Number of samples to process at the same time. Defaults to 1.
```

4. After the analysis, run `segments.py` to extract short audio segments for species detections to verify results. This way, it might be easier to review results instead of loading hundreds of result files manually.

Here's a complete list of all command line arguments:

```
--audio, Path to folder containing audio files.
--results, Path to folder containing result files.
--o, Output folder path for extracted segments.
--min_conf, Minimum confidence threshold. Values in [0.01, 0.99]. Defaults to 0.1.
--max_segments, Number of randomly extracted segments per species.
--seg_length, Length of extracted segments in seconds. Defaults to 3.0.
--threads, Number of CPU threads.
```

5. When editing your own `species_list.txt` file, make sure to copy species names from the labels file of each model. 

You can find label files in the checkpoints folder, e.g., `checkpoints/V2.1/BirdNET_GLOBAL_2K_V2.1_Labels.txt`. 

Species names need to consist of `scientific name_common name` to be valid.

6. You can generate a species list for a given location using `species.py` in case you need it for reference. Here is an example:

```
python3 species.py --o example/species_list.txt --lat 42.5 --lon -76.45 --week 4
```

Here's a complete list of all command line arguments:

```
--o, Path to output file or folder. If this is a folder, file will be named 'species_list.txt'.
--lat, Recording location latitude. Set -1 to ignore.
--lon, Recording location longitude. Set -1 to ignore.
--week, Week of the year when the recording was made. Values in [1, 48] (4 weeks per month). Set -1 for year-round species list.
--threshold, Occurrence frequency threshold. Defaults to 0.05.
--sortby, Sort species by occurrence frequency or alphabetically. Values in ['freq', 'alpha']. Defaults to 'freq'.
```

7. This is a very basic version of the analysis workflow, you might need to adjust it to your own needs.

8. Please open an issue to ask for new features or to document unexpected behavior.

9. I will keep models up to date and upload new checkpoints whenever there is an improvement in performance. I will also provide quantized and pruned model files for distribution.

## Usage (Docker)

Install docker for Ubuntu:

```
sudo apt install docker.io
```

Build Docker container:

```
sudo docker build -t birdnet .
```

<b>NOTE</b>: You need to run docker build again whenever you make changes to the script.

In order to pass a directory that contains your audio files to the docker file, you need to mount it inside the docker container with <i>-v /my/path:/mount/path</i> before you can run the container. 

You can run the container for the provided example soundscapes with:

```
sudo docker run -v $PWD/example:/audio birdnet --i audio --o audio --slist audio
```

You can adjust the directory that contains your recordings by providing an absolute path:

```
sudo docker run -v /path/to/your/audio/files:/audio birdnet --i audio --o audio --slist audio
```

You can also mount more than one drive, e.g., if input and output folder should be different:

```
sudo docker run -v /path/to/your/audio/files:/input -v /path/to/your/output/folder:/output birdnet --i input --o output --slist input
```

See "Usage" section above for more command line arguments, all of them will work with Docker version.

<b>NOTE</b>: If you like to specify a species list (which will be used as post-filter and needs to be named 'species_list.txt'), you need to put it into a folder that also has to be mounted. 

## Usage (Server)

You can host your own analysis service and API by launching the `server.py` script. This will allow you to send files to this server, store submitted files, analyze them and send detection results back to a client. This could be a local service, running on a desktop PC, or a remote server. The API can be accessed locally or remotely through a browser or Python client (or any other client implementation).

1. Install one additional package with `pip3 install bottle`.

2. Start the server with `python3 server.py`. You can also specify a host name or IP and port number, e.g., `python3 server.py --host localhost --port 8080`.

Here's a complete list of all command line arguments:

```
--host, Host name or IP address of API endpoint server. Defaults to '0.0.0.0'.
--port, Port of API endpoint server. Defaults to 8080.    
--spath, Path to folder where uploaded files should be stored. Defaults to '/uploads'.
--threads, Number of CPU threads for analysis. Defaults to 4.
--locale, Locale for translated species common names. Values in ['af', 'de', 'it', ...] Defaults to 'en'.
```

<b>NOTE</b>: The server is single-threaded, so you'll need to start multiple instances for higher throughput. This service is intented for short audio files (e.g., 1-10 seconds).

3. Query the API with a client. You can use the provided Python client or any other client implementation. Request payload needs to be `multipart/form-data` with the following fields: `audio` for raw audio data as byte code, and `meta` for additional information on the audio file. Take a look at our example client implementation in the `client.py` script.

This script will read an audio file, generate metadata from command line arguments and send it to the server. The server will then analyze the audio file and send back the detection results which will be stored as a JSON file.

Here's a complete list of all command line arguments:

```
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
```

4. Parse results from the server. The server will send back a JSON response with the detection results. The response also contains a `msg` field, indicating `success` or `error`. Results consist of a sorted list of (species, score) tuples.

This is an example response:

```
{"msg": "success", "results": [["Poecile atricapillus_Black-capped Chickadee", 0.7889], ["Spinus tristis_American Goldfinch", 0.5028], ["Junco hyemalis_Dark-eyed Junco", 0.4943], ["Baeolophus bicolor_Tufted Titmouse", 0.4345], ["Haemorhous mexicanus_House Finch", 0.2301]]}
```

<b>NOTE</b>: Let us know if you have any questions, suggestions, or feature requests. Also let us know when hosting an analysis service - we would love to give it a try.

## Usage (GUI)

We provide a very basic GUI which lets you launch the analysis through a web interface. 

![GUI screenshot](https://tuc.cloud/index.php/s/QyBczrWXCrMoaRC/download/analyzer_gui.png)

1. You need to install one additional package in order to use the GUI with `pip3 install pywebview`
2. Launch the GUI with `python3 gui.py`.
3. Set all folders and parameters, after that, click 'Start analysis'. 

Status updates should be visible in 'Status' text area.

<b>NOTE</b>: You can easily adjust the interface by editing `gui/index.html` and `gui/style.css`. Feel free to submit your udated (possibly better looking) version through a pull request. 

## Funding

This project is supported by Jake Holshuh (Cornell class of ’69) and The Arthur Vining Davis Foundations. Our work in the K. Lisa Yang Center for Conservation Bioacoustics is made possible by the generosity of K. Lisa Yang to advance innovative conservation technologies to inspire and inform the conservation of wildlife and habitats.

The European Union and the European Social Fund for Germany partially funded this research. This work was also partially funded by the German Federal Ministry of Education and Research in the program of Entrepreneurial Regions InnoProfileTransfer in the project group localizeIT (funding code 03IPT608X).

## Partners

BirdNET is a joint effort of partners from academia and industry. Without these partnerships, this project would not have been possible. Thank you!

![Logos of all partners](https://tuc.cloud/index.php/s/KSdWfX5CnSRpRgQ/download/box_logos.png)
