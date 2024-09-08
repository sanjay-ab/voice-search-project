"""Use to resample audio files to 16kHz."""
import librosa
import soundfile as sf
import os

if __name__ == "__main__":
    language = "odia"

    print(f"Downsampling {language} dataset")

    for folder in ["test", "train"]:
        print(f"Downsampling {folder} set")

        directory = f"{language}/{folder}/audio"
        out_directory = f"{language}_upsampled/{folder}/audio"
        os.makedirs(out_directory, exist_ok=True)
        files = os.listdir(directory)
        percentage_done = 0
        total_length = len(files)
        for i, file in enumerate(files):
            file_path = os.path.join(directory, file)
            y, sr = librosa.load(file_path, sr=8000)
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)

            output_file_path = os.path.join(out_directory, file)
            sf.write(output_file_path, y, 16000)

            if i / total_length * 100 > percentage_done:
                percentage_done += 10
                print(f"{percentage_done}% done")