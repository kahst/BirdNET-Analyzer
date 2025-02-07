Command line interface
======================

birdnet_analyzer.analyze
------------------------

.. argparse::
   :ref: birdnet_analyzer.cli.analyzer_parser
   :prog: birdnet_analyzer.analyze

birdnet_analyzer.client
------------------------

.. argparse::
   :ref: birdnet_analyzer.cli.client_parser
   :prog: birdnet_analyzer.client
   :markdown:

   This script will read an audio file, generate metadata from command line arguments and send it to the server.
   The server will then analyze the audio file and send back the detection results which will be stored as a JSON file.

birdnet_analyzer.embeddings
------------------------

.. argparse::
   :ref: birdnet_analyzer.cli.embeddings_parser
   :prog: birdnet_analyzer.embeddings

birdnet_analyzer.segments
-------------------------

.. argparse::
   :ref: birdnet_analyzer.cli.segments_parser
   :prog: birdnet_analyzer.segments

birdnet_analyzer.species
-------------------------

.. argparse::
   :ref: birdnet_analyzer.cli.species_parser
   :prog: birdnet_analyzer.species

birdnet_analyzer.server
-------------------------

.. argparse::
   :ref: birdnet_analyzer.cli.server_parser
   :prog: birdnet_analyzer.server
   :markdown:

   You can host your own analysis service and API by launching the `birdnet_analyzer.server` script.
   This will allow you to send files to this server, store submitted files, analyze them and send detection results back to a client.
   This could be a local service, running on a desktop PC, or a remote server.
   The API can be accessed locally or remotely through a browser or Python client (or any other client implementation).

   Install one additional package with `pip install bottle`.

   Start the server with `python -m birdnet_analyzer.server`.
   You can also specify a host name or IP and port number, e.g., `python -m birdnet_analayzer.server --host localhost --port 8080`.

   The server is single-threaded, so you’ll need to start multiple instances for higher throughput. This service is intented for short audio files (e.g., 1-10 seconds).

   Query the API with a client.
   You can use the provided Python client or any other client implementation.
   Request payload needs to be `multipart/form-data` with the following fields:
   `audio` for raw audio data as byte code, and `meta` for additional information on the audio file.
   Take a look at our example client implementation in the `client.py` script.

   Parse results from the server. The server will send back a JSON response with the detection results. The response also contains a msg field, indicating success or error. Results consist of a sorted list of (species, score) tuples.

   This is an example response:

   ```json
   {"msg": "success", "results": [["Poecile atricapillus_Black-capped Chickadee", 0.7889], ["Spinus tristis_American Goldfinch", 0.5028], ["Junco hyemalis_Dark-eyed Junco", 0.4943], ["Baeolophus bicolor_Tufted Titmouse", 0.4345], ["Haemorhous mexicanus_House Finch", 0.2301]]}
   ````

birdnet_analyzer.train
-------------------------

.. argparse::
   :ref: birdnet_analyzer.cli.train_parser
   :prog: birdnet_analyzer.train