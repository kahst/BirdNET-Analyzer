from birdnet_analyzer import audio

def test_get_sorted_timestamps():
    len = audio.getAudioFileLength('birdnet_analyzer/example/soundscape.wav', 48000)
    assert len == 120.0 , "known audio file `soundscape.wav` should have length 120.0"