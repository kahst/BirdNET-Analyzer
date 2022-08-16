## BirdNET Analyzer Model Version History

This repository contains only the latest version of the BirdNET-Analyzer model. We will track the changes in this document and also provide permanent download links to previous versions for testing and/or other use cases. 

You can download and unzip previous model versions, make sure to update the `MODEL_PATH` and `LABELS_FILE` variable in the `config.py` file to the version you want to use.

Model update history:

**V2.2**

- smaller (21.3 MB vs. 29.5 MB as FP32) and faster (1.31 vs 2.03 GFLOPs) than V2.1
- maintains same accuracy as V2.1 despite more classes
- global selection of species (birds and non-birds) with 3,337 classes (incl. 10 non-event classes)

**V2.1**

- same model architecture as V2.0
- extended 2022 training data
- global selection of species (birds and non-birds) with 2,434 classes (incl. 10 non-event classes)
- Download here: [BirdNET-Analyzer-V2.1.zip](https://drive.google.com/file/d/15cvPiezn_6H2tQs1FGMVrVdqiwLjLRms)

**V2.0**

- same model design as 1.4 but a bit wider
- extended 2022 training data
- global selection of species (birds and non-birds) with 1,328 classes (incl. 10 non-event classes)
- Download here: [BirdNET-Analyzer-V2.0.zip](https://drive.google.com/file/d/1h2Tbk_29ghNdK62ynrdRWyxT4H1fpFGs)

**V1.4**

- smaller, deeper, faster
- only 30% of the size of V1.3
- still linear spectrogram and EfficientNet blocks
- extended 2021 training data
- 1,133 birds and non-birds for North America and Europe
- Download here: [BirdNET-Analyzer-V1.4.zip](https://drive.google.com/file/d/1h14-Y8dOrPr9XCWfIoUjlWMJ9aWyNkKa)

**V1.3**

- Model uses linear frequency scale for spectrograms
- uses V2 fusion blocks and V1 efficient blocks
- extended 2021 training data
- 1,133 birds and non-birds for North America and Europe
- Download here: [BirdNET-Analyzer-V1.3.zip](https://drive.google.com/file/d/1h0nJzPjyJWbkfPyaWpS332xUwzDOygs9)

**V1.2**

- Model based on EfficientNet V2 blocks
- uses V2 fusion blocks and V1 efficient blocks
- extended 2021 training data
- 1,133 birds and non-birds for North America and Europe
- Download here: [BirdNET-Analyzer-V1.2.zip](https://drive.google.com/file/d/1h-il_W6t8Tz_XHrRMO1zcp_ThYp9QPLK)

**V1.1**

- Model based on Wide-ResNet (aka "App model")
- extended 2021 training data
- 1,133 birds and non-birds for North America and Europe
- Download here: [BirdNET-Analyzer-V1.1.zip](https://drive.google.com/file/d/1gzpwiCAf2HkfcAmlRq1K9Q0KrDsd5nGP)

**App Model**

- Model based on Wide-ResNet
- ~3,000 species worldwide
- currently deployed as BirdNET app model
- Download here: [BirdNET-Analyzer-App-Model.zip](https://drive.google.com/file/d/1gxkxPFlaTYxHFqAODDHYGUX8uEkZDWaL)