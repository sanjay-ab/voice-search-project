from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import os
from utils.common_functions import read_wav_file


class AudioDataset(Dataset):
    def __init__(self, audio_directory):
        self.audio_files = sorted(os.listdir(audio_directory))
        self.audio_directory = audio_directory

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        fname = self.audio_files[idx]
        audio_file_path = os.path.join(self.audio_directory, fname)
        speech = read_wav_file(audio_file_path)

        return speech, fname, idx

def collate_fn(batch):

    max_length = 0
    fnames = [0] * len(batch)
    idxs = [0] * len(batch)
    speech_list = [0] * len(batch)
    for i, tup in enumerate(batch):
        speech, fname, idx = tup
        fnames[i] = fname
        idxs[i] = idx
        speech_list[i] = speech
        max_length = max(max_length, len(speech))

    return speech_list, fnames, idxs, max_length

if __name__ == "__main__":
    dataset = AudioDataset("data/tamil/all_data")
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    speech, fname, idx, max_length = next(iter(dataloader))
    # length = len(dataloader)
    # for speech, fname, idxs in dataloader:
    #     percentage = idxs[-1]/length * 100
    #     print(f"{percentage:.2f}% done")
    #     print(speech.shape, fname, idxs)
    # print(speech.shape, fname)
    print(speech[0].shape, fname[0])
    print(speech)
    print(idx)