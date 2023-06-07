from pydub import AudioSegment

#sys.path.append('c:\python39\lib\site-packages\ffmpeg-python')
file_probe = r"C:\Users\willfrid boris\Documents\william\ffmpeg-6.0-essentials_build\bin\ffprobe.exe"

file_mpeg = r"C:\Users\willfrid boris\Documents\william\ffmpeg-6.0-essentials_build\bin\ffmpeg.exe"

# pydub.utils.(file_pat)
AudioSegment.converter = file_mpeg
AudioSegment.ffmpeg = file_mpeg
AudioSegment.ffprobe = file_probe
