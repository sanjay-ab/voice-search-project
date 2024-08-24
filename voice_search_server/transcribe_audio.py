import voice_search_server.lib.asr as asr
# import librosa
import soundfile as sf
import os

if __name__ == '__main__':
    model = asr.create_asr('model')
    path = "../data/digit_recog_wavs_downsampled"
    for file in sorted(os.listdir(path)):
        print(file)
        # audio, _ = librosa.load(f"{path}/{file}", mono=True, sr=16000)
        audio, _ = sf.read(path, dtype='float32')
        pcm = (audio * 32767).astype('<u2').tobytes()
        print(model.transcribe(pcm))