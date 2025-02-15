FAQ
===

We will answer frequently asked questions here. If you have a question that is not answered here, please let us know at `ccb-birdnet@cornell.edu <mailto:ccb-birdnet@cornell.edu>`_.

What is BirdNET-Analyzer?
-------------------------

BirdNET-Analyzer is a tool for analyzing bird sounds using machine learning models. It can identify bird species from audio recordings and provides various functionalities for training custom classifiers, extracting segments, and reviewing results.

How do I install BirdNET-Analyzer?
----------------------------------

BirdNET-Analyzer can be installed using different methods, including:

- | **Raven Pro**: Follow the instructions provided in the Raven Pro documentation.
- | **Python Package**: Install via pip using `pip install birdnet-analyzer`.
- | **Command Line**: Download the repository and run the scripts from the command line.
- | **GUI**: Download the GUI version from the website and follow the installation instructions.

What licenses are used in BirdNET-Analyzer?
-------------------------------------------

BirdNET-Analyzer source code is released under the MIT License. The models used in the project are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0). Please review and adhere to the specific license terms provided with each model.
Custom models trained with BirdNET-Analyzer are also subject to the same licensing terms.

.. note:: Please note that all educational and research purposes are considered non-commercial use and it is therefore freely permitted to use BirdNET models in any way.

Please get in touch if you have any questions or need further assistance. 

How do I create a custom species list?
--------------------------------------

To create a custom species list, follow these steps:

- | Copy species names from the labels file of each model, found in the checkpoints folder (e.g., `checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels.txt`).
- | Ensure species names are in the format `scientific name_common name`.
- | Use the `species.py` script to generate a species list for a given location and time.

How do I use the "segments" function in the GUI?
------------------------------------------------

The "segments" function in the GUI allows you to create a collection of species-specific predictions that exceed a user-defined confidence value. To use this function:

- | Select audio, result, and output directories.
- | Set additional parameters such as the minimum confidence value, the maximum number of segments per species, the audio speed, and the segment length.
- | Start the extraction process. BirdNET will create subfolders for each identified species and save audio clips of the corresponding recordings.

How do I review the extracted segments?
---------------------------------------

The review tab in the GUI allows you to systematically review and label the extracted segments. It provides tools for visualizing spectrograms, listening to audio segments, and categorizing them as positive or negative detections. You can also generate logistic regression plots to visualize the relationship between confidence values and the likelihood of correct detections.

What are BirdNET confidence values?
-----------------------------------

BirdNET confidence values are a measure of the algorithm's prediction reliability. They are not probabilities and are not directly transferable between different species or recording conditions. It is recommended to start with the highest confidence scores and work down to the lower scores when reviewing results.

What are the non-event classes in BirdNET?
------------------------------------------

There are currently 11 non-event classes in BirdNET:

* Human non-vocal_Human non-vocal
* Human vocal_Human vocal
* Human whistle_Human whistle
* Noise_Noise
* Dog_Dog
* Engine_Engine
* Environmental_Environmental
* Fireworks_Fireworks
* Gun_Gun
* Power tools_Power tools
* Siren_Siren

`Noise_Noise` and `Environmental_Environmental` are auxiliary classes used for training and will never be predicted by the model.