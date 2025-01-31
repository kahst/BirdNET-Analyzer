Models
======

Hier kommt Model version update hin, die Technical Details und die Anleitung wie man ältere Modelle verwendet.


Model Version History
---------------------

V2.4
^^^^

- more than 6,000 species worldwide
- covers frequencies from 0 Hz to 15 kHz with two-channel spectrogram (one for low and one for high frequencies)
- 0.826 GFLOPs, 50.5 MB as FP32
- enhanced and optimized metadata model
- global selection of species (birds and non-birds) with 6,522 classes (incl. 10 non-event classes)
- Download here: `BirdNET-Analyzer-V2.4.zip <https://drive.google.com/file/d/1ixYBPbZK2Fh1niUQzadE2IWTFZlwATa3>`_

V2.3
^^^^

- slightly larger (36.4 MB vs. 21.3 MB as FP32) but smaller computational footprint (0.698 vs. 1.31 GFLOPs) than V2.2
- larger embedding size (1024 vs 320) than V2.2 (hence the bigger model)
- enhanced and optimized metadata model
- global selection of species (birds and non-birds) with 3,337 classes (incl. 10 non-event classes)
- Download here: `BirdNET-Analyzer-V2.3.zip <https://drive.google.com/file/d/1hhwQBVBngGnEhmqYeDksIW8ZY1FJmwyi>`_

V2.2
^^^^

- smaller (21.3 MB vs. 29.5 MB as FP32) and faster (1.31 vs 2.03 GFLOPs) than V2.1
- maintains same accuracy as V2.1 despite more classes
- global selection of species (birds and non-birds) with 3,337 classes (incl. 10 non-event classes)
- Download here: `BirdNET-Analyzer-V2.2.zip <https://drive.google.com/file/d/166w8IAkXGKp6ClKb8vaniG1DmOr8Fwem>_

V2.1
^^^^

- same model architecture as V2.0
- extended 2022 training data
- global selection of species (birds and non-birds) with 2,434 classes (incl. 10 non-event classes)
- Download here: `BirdNET-Analyzer-V2.1.zip <https://drive.google.com/file/d/15cvPiezn_6H2tQs1FGMVrVdqiwLjLRms>`_

V2.0
^^^^

- same model design as 1.4 but a bit wider
- extended 2022 training data
- global selection of species (birds and non-birds) with 1,328 classes (incl. 10 non-event classes)
- Download here: `BirdNET-Analyzer-V2.0.zip <https://drive.google.com/file/d/1h2Tbk_29ghNdK62ynrdRWyxT4H1fpFGs>`_

V1.4
^^^^

- smaller, deeper, faster
- only 30% of the size of V1.3
- still linear spectrogram and EfficientNet blocks
- extended 2021 training data
- 1,133 birds and non-birds for North America and Europe
- Download here: `BirdNET-Analyzer-V1.4.zip <https://drive.google.com/file/d/1h14-Y8dOrPr9XCWfIoUjlWMJ9aWyNkKa>`_

V1.3
^^^^

- Model uses linear frequency scale for spectrograms
- uses V2 fusion blocks and V1 efficient blocks
- extended 2021 training data
- 1,133 birds and non-birds for North America and Europe
- Download here: `BirdNET-Analyzer-V1.3.zip <https://drive.google.com/file/d/1h0nJzPjyJWbkfPyaWpS332xUwzDOygs9>`_

V1.2
^^^^

- Model based on EfficientNet V2 blocks
- uses V2 fusion blocks and V1 efficient blocks
- extended 2021 training data
- 1,133 birds and non-birds for North America and Europe
- Download here: `BirdNET-Analyzer-V1.2.zip <https://drive.google.com/file/d/1h-il_W6t8Tz_XHrRMO1zcp_ThYp9QPLK>`_

V1.1
^^^^

- Model based on Wide-ResNet (aka "App model")
- extended 2021 training data
- 1,133 birds and non-birds for North America and Europe
- Download here: `BirdNET-Analyzer-V1.1.zip <https://drive.google.com/file/d/1gzpwiCAf2HkfcAmlRq1K9Q0KrDsd5nGP>`_

App Model
^^^^^^^^^

- Model based on Wide-ResNet
- ~3,000 species worldwide
- currently deployed as BirdNET app model
- Download here: `BirdNET-Analyzer-App-Model.zip <https://drive.google.com/file/d/1gxkxPFlaTYxHFqAODDHYGUX8uEkZDWaL>`_
