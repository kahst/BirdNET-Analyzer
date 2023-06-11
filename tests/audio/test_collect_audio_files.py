from typing import List
from birdnet.audio.audio_files_collection import collect_audio_files


def test_collect_audio_files():
    path = 'example'
    audio_files = collect_audio_files(path=path)

    assert audio_files is not None
    assert isinstance(audio_files, List)
    assert audio_files != []
