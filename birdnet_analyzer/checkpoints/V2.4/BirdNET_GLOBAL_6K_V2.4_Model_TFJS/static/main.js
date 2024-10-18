// Simple status function to print messages to the page
function setStatus(status, new_line = true) {
    document.getElementById('status').innerHTML += status;
    if (new_line) {
        document.getElementById('status').innerHTML += '<br>';
    }
    console.log(status);
}

// Define custom layer for computing mel spectrograms
class MelSpecLayerSimple extends tf.layers.Layer {
    constructor(config) {
        super(config);

        // Initialize parameters
        this.sampleRate = config.sampleRate;
        this.specShape = config.specShape;
        this.frameStep = config.frameStep;
        this.frameLength = config.frameLength;
        this.fmin = config.fmin;
        this.fmax = config.fmax;
        this.melFilterbank = tf.tensor2d(config.melFilterbank);
    }

    build(inputShape) {
        // Initialize trainable weights, for example:
        this.magScale = this.addWeight(
            'magnitude_scaling',
            [],
            'float32',
            tf.initializers.constant({ value: 1.23 })
        );

        super.build(inputShape);
    }

    // Compute the output shape of the layer
    computeOutputShape(inputShape) {
        return [inputShape[0], this.specShape[0], this.specShape[1], 1];
    }

    // Define the layer's forward pass
    call(inputs) {
        return tf.tidy(() => {
            // inputs is a tensor representing the input data
            inputs = inputs[0];
            // Split 'inputs' along batch dimension into array of tensors with length == batch size
            const inputList = tf.split(inputs, inputs.shape[0])
            // Perform STFT on each tensor in the array
            const specBatch = inputList.map(input =>{
                input = input.squeeze();
                // Normalize values between -1 and 1
                input = tf.sub(input, tf.min(input, -1, true));
                input = tf.div(input, tf.max(input, -1, true).add(0.000001));
                input = tf.sub(input, 0.5);
                input = tf.mul(input, 2.0);

                // Perform STFT
                let spec = tf.signal.stft(
                    input,
                    this.frameLength,
                    this.frameStep,
                    this.frameLength,
                    tf.signal.hannWindow,
                );

                // Cast from complex to float
                spec = tf.cast(spec, 'float32');

                // Apply mel filter bank
                spec = tf.matMul(spec, this.melFilterbank);

                // Convert to power spectrogram
                spec = spec.pow(2.0);

                // Apply nonlinearity
                spec = spec.pow(tf.div(1.0, tf.add(1.0, tf.exp(this.magScale.read()))));

                // Flip the spectrogram
                spec = tf.reverse(spec, -1);

                // Swap axes to fit input shape
                spec = tf.transpose(spec)

                // Adding the channel dimension
                spec = spec.expandDims(-1);

                return spec;
            })
            // Convert tensor array into batch tensor
            return tf.stack(specBatch)
        });
    }

    // Optionally, include the `className` method to provide a machine-readable name for the layer
    static get className() {
        return 'MelSpecLayerSimple';
    }
}

// Register the custom layer with TensorFlow.js
tf.serialization.registerClass(MelSpecLayerSimple);

// Main function
async function run() {
    await tf.ready();  // Make sure TensorFlow.js is fully loaded

    // Load model with custom layer (this can take a while the first time)
    setStatus('Loading model...', false);
    const model = await tf.loadLayersModel('static/model/model.json', {custom_objects: {'MelSpecLayerSimple': MelSpecLayerSimple}});
    setStatus('Done!')

    // Load labels
    setStatus('Loading labels...', false);
    const label_data = await fetch('static/model/labels.json');
    const labels = await label_data.json();
    setStatus('Done!')

    // Load the audio file
    setStatus('Loading audio file...', false);
    const response = await fetch('static/sample.wav');
    const arrayBuffer = await response.arrayBuffer();

    // Decode the audio data 
    // - we need to set sampleRate option to prevent AudioContext resampling the file to the its default 44100Hz
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000});
    const audioBuffer = await new Promise((resolve, reject) => {
        audioCtx.decodeAudioData(arrayBuffer, resolve, reject);
    });

    // Read audio data
    const audioData = audioBuffer.getChannelData(0); // Get data for the first channel

    // TODO: Resample and split audio data into chunks of 3 seconds
    // to match the input shape of the model (1, 144000)
    // For now, let's use a sample that is already 3 seconds long and sampled at 48 kHz

    // Create a tensor from the audio data
    // Only works for batch size 1 for now
    let input = tf.tensor(audioData).reshape([1, 144000]);
    setStatus('Done!')

    // Run the prediction
    setStatus('Running prediction...', false);
    const prediction = model.predict(input);
    setStatus('Done!')

    // Print top 3 probabilities and labels
    setStatus('<b>Results:</b>');
    const probs = await prediction.data();
    const probs_sorted = probs.slice().sort().reverse();
    for (let i = 0; i < 3; i++) {
        const index = probs.indexOf(probs_sorted[i]);
        setStatus(labels[index] + ': ' + probs_sorted[i]);
    }

    // Load metadata model
    setStatus('<br>Loading metadata model...', false);
    const metadata_model = await tf.loadGraphModel('static/model/mdata/model.json');
    setStatus('Done!')

    // Dummy location and week
    const lat = 52.5;
    const lon = 13.4;
    const week = 42;
    let mdata_input = tf.tensor([lat, lon, week]).expandDims(0);

    // Run the prediction
    setStatus('Running mdata prediction...', false);
    const mdata_prediction = metadata_model.predict(mdata_input);
    setStatus('Done!')

    // Print top 10 probabilities and labels (labels are the same as for the audio model)
    setStatus('<b>Most common species @ (' + lat + '/' + lon + ') in week ' + week + ':</b>');
    const mdata_probs = await mdata_prediction.data();
    const mdata_probs_sorted = mdata_probs.slice().sort().reverse();
    for (let i = 0; i < 10; i++) {
        const index = mdata_probs.indexOf(mdata_probs_sorted[i]);
        setStatus(labels[index] + ': ' + mdata_probs_sorted[i]);
    }

}

// Run the function above after the page is fully loaded
// To avoid CORS security errors, the run function must be called after a user interaction
// or the AudioContext will be blocked, however, let's only enable the button when the page is loaded

window.addEventListener('load', () => {
    const button = document.getElementById('button')
    button.value = 'Ready. Click to run example';
    button.disabled = false;
});
