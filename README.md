
<style type="text/css">
  body {
    background-color:  #fdf6e3;
    color: #268bd2;
  }
  p {
    background-color:  #fdf6e3;
    color: #586e75;
  }
  ul {
    background-color:  #fdf6e3;
    color: #586e75;
  }
  code {
    background-color:  #93a1a1;
    color: #6c71c4 ;
  }

</style>

<h1 style="text-align: center;">BattyBirdNET-Analyzer</h1>


## Purpose

The purpose is to assist in the automated classification of bat calls. There are some excellent classifiers out there. However, most are for other animal groups, notably birds, or are limited in their use to researchers. The aim here is to provide a solid classifier for as many regions of the planet as possible that can be built into inexpensive systems for citizen scientists. It builds upon the work of the BirdNET open source product family that has a similar aim for birds. The BirdNET systems are using a sampling frequency that is too low to capture bat echo location calls. So here I use the system at significantly higher sampling rates and cross train a classifier on the BirdNET artificial neural networks to identify bats. Think of a bat as a bird singing three times as fast. Once such a cross trained bat network has the necessary performance, I intend to integrate it into the BirdNET-Pi framework in order to run them 24/7 in self-assembled recoding stations.
You can most definitely already use the bat trained classifier with the analyzer scripts to assist in identifying bat calls.
Currently includes the following species:

## Species covered
#### North America

- Antrozous pallidus ( Pallid bat )
- Corynorhinus rafinesquii ( Rafinesque's big-eared bat ) 
- Corynorhinus townsendii ( Townsend's big-eared bat )
- Eptesicus fuscus ( Big brown bat )
- Euderma maculatum ( Spotted bat )
- Eumops floridanus ( Florida bonneted )
- Eumops perotis ( Greater mastiff bat )
- Idionycteris phyllotis ( Allen's big-eared bat )
- Lasionycteris noctivagans ( Silver-haired bat )
- Lasiurus blossevillii ( Western red bat )
- Lasiurus borealis ( Eastern red bat )
- Lasiurus cinereus ( Hoary bat )
- Lasiurus intermedius ( Northern yellow bat )
- Lasiurus seminolus ( Seminole bat )
- Lasiurus xanthinus ( Western yellow bat )
- Myotis austroriparius ( Southeastern myotis )
- Myotis californicus ( California bat )
- Myotis ciliolabrum ( Western small-footed myotis )
- Myotis evotis ( Western long-eared bat )
- Myotis grisescens ( Gray bat )
- Myotis leibii ( Eastern small-footed bat )
- Myotis lucifugus ( Little brown bat )
- Myotis septentrionalis ( Northern long-eared bat )
- Myotis sodalis ( Indiana bat )
- Myotis thysanodes ( Fringed bat )
- Myotis velifer ( Cave bat )
- Myotis volans ( Long-legged bat )
- Myotis yumanensis ( Yuma bat )
- Nycticeius humeralis ( Evening bat )
- Nyctinomops femorosaccus ( Pocketed free-tailed bat )
- Nyctinomops macrotis ( Big free-tailed bat )
- Parastrellus hesperus ( Canyon bat )
- Perimyotis subflavus ( Tricolored bat )
- Tadarida brasiliensis ( Brazilian free-tailed bat)`

LICENSE: http://creativecommons.org/licenses/by-nc-sa/4.0/

## Classifiers
The available classifiers include:

- **"nabat-100-144kHz-200epochs.tflite"** which has been trained on max. 100 calls from all the species in the NABAT data set. It has not been formally evaluated (as yet), yet the training stopped when:
``` sh
loss: 0.0034 - prec: 0.9564 - val_loss: 0.0032 - val_prec: 0.9631
```

## Usage
In principle all the scripts inherited from BirdNET-Analyzer can work. Since you do not want to analyze bird calls, you will have to provide the necessary command line parameter. Also, at this time the georgaphical lat long parameters ar not (yet) enabled.

- Inspect config file for options and settings, especially inference settings. 
- If you do change the sampling_rate settings, note that only SIG_LENGTH *SAMPLING_RATE = 144000 combinations will work. 
- You need to set paths for the audio file and selection table output.
- You call the scripts from within the top directory BattyBirdNET-Analyzer
Here is an example that looks to identify bats in the NABAT set:
``` sh
python3 analyze.py --classifier ./checkpoints/bats/nabat-100-144kHz-200epochs.tflite --i /path/to/audio/folder --o /path/to/output/folder
```

NOTE: You can also specify the number of CPU threads that should be used for the analysis with `--threads <Integer>` (e.g., `--threads 16`).
Do not use the  `--lat` and `--lon` or it will look for birds at this time.

You can test the setup with

``` sh
python3 analyze.py --classifier ./checkpoints/bats/nabat-100-144kHz-200epochs.tflite --i ./example/MYAU-57238183.wav --o ./example/MYAU-57238183.txt
```

or for the entire example directory

``` sh
python3 analyze.py --classifier ./checkpoints/bats/nabat-100-144kHz-200epochs.tflite --i ./example/ --o ./example/
```
Here's a complete list of all command line arguments that make sense for bats right now:

``` 
--i, Path to input file or folder. If this is a file, --o needs to be a file too.
--o, Path to output file or folder. If this is a file, --i needs to be a file too.
--sensitivity, Detection sensitivity; Higher values result in higher sensitivity. Values in [0.5, 1.5]. Defaults to 1.0.
--min_conf, Minimum confidence threshold. Values in [0.01, 0.99]. Defaults to 0.1.
--overlap, Overlap of prediction segments. Values in [0.0, 0.9]. Defaults to 0.0.
--rtype, Specifies output format. Values in ['table', 'audacity', 'r', 'kaleidoscope', 'csv']. Defaults to 'table' (Raven selection table).
--threads, Number of CPU threads.
--batchsize, Number of samples to process at the same time. Defaults to 1.
-classifier, Path to custom trained classifier. Defaults to None. If set, --lat, --lon and --locale are ignored.
```

## Install


## Methods and data


## References and thanks


<p align="center">
  <img src="./assets/BattyBirdNET-logo--.png" />
</p>
