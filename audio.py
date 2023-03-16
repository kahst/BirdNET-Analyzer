import numpy as np

import config as cfg
import soundfile as sf
import torchaudio.transforms as T
import torch
import torchaudio
import numpy
import datetime

RANDOM = np.random.RandomState(cfg.RANDOM_SEED)

inputSampleRate = 24000
def openAudioFile(path, sample_rate=48000, offset=0.0, duration=None):    
    
    # Open file with librosa (uses ffmpeg or libav)
    import librosa

    try:
        fileSampleRate = librosa.get_samplerate(path)
        #print(fileSampleRate)
        #torch.set_num_threads(1)
        #sig, rate = librosa.load(path, sr=sample_rate, offset=offset, duration=duration, mono=True, res_type='kaiser_fast')#kaiser_best,'kaiser_fast'
        waveform, sample_rate = torchaudio.load(path, frame_offset=offset*fileSampleRate, num_frames=duration*fileSampleRate)
        defaultRolloff= 0.9475937167399596
        #.6 seems to be around 3700hz
        #.5 filters at 3000 use as a DEFAULT, seems to greatly increase detectability
        #.3 might be ideal for GGOW filtering
        #.125 might work well to filter frogs from the audio
        resampler = T.Resample(fileSampleRate, 48000, dtype=waveform.dtype, lowpass_filter_width=64,rolloff=0.3,resampling_method="kaiser_window",beta=14.769656459379492)
        sig = resampler(waveform).numpy()[0]
        rate = 48000
    except:
        sig, rate = [], sample_rate

    return sig, rate

def openAudioFileNoResample(path, sample_rate=48000, offset=0.0, duration=None):    
    
    # Open file with librosa (uses ffmpeg or libav)
    import librosa

    try:
        sig, rate = librosa.load(path, sr=None, offset=offset, duration=duration, mono=True)#kaiser_best,'kaiser_fast'

    except:
        sig, rate = [], sample_rate

    return sig, rate

def getAudioFileLength(path, sample_rate=48000, offset=0.0, duration=None):    
    
    # Open file with librosa (uses ffmpeg or libav)
    import librosa

    return librosa.get_duration(filename=path, sr=sample_rate)

def saveSignal(sig, fname, sampleRate = 48000):

    import soundfile as sf
    sf.write(fname, sig, sampleRate, 'PCM_16')

def noise(sig, shape, amount=None):

    # Random noise intensity
    if amount == None:
        amount = RANDOM.uniform(0.1, 0.5)

    # Create Gaussian noise
    try:
        noise = RANDOM.normal(min(sig) * amount, max(sig) * amount, shape)
    except:
        noise = np.zeros(shape)

    return noise.astype('float32')

def splitSignal(sig, rate, seconds, overlap, minlen):

    # Split signal with overlap
    sig_splits = []
    for i in range(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + int(seconds * rate)]
        # End of signal?
        if len(split) < int(minlen * rate):
            break
        
        # Signal chunk too short?
        if len(split) < int(rate * seconds):
            split = np.hstack((split, noise(split, (int(rate * seconds) - len(split)), 0.5)))
        
        sig_splits.append(split)

    return sig_splits