from birdnet.utils.audio_file_collecting import collect_audio_files

from tests._paths import ROOT_PATH


def test_collect_audio_files():
    path = 'example'
    audio_files = collect_audio_files(path=str(ROOT_PATH / path))

    assert audio_files is not None
    assert isinstance(audio_files, list)
    assert audio_files != []
