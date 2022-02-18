from analyze_adapted import *

wiwa_dirs = Path("example")
dest_dir = Path("example")
try:
  os.mkdir(dest_dir)
except:
  pass

# list all files
wav_paths = Path(wiwa_dirs).rglob('*.wav')

for wav_file in wav_paths:
  analyze_files(wav_file, dest_dir)

# analyze_files(Path('/media/oekofor/Daten_1/002_WiWa2/Daten/Aufnahmen/SN001_audio/SN001_2019_02_21_17_53_07_evening.wav'), dest_dir)
