"""Extract mHuBERT embeddings from wav files."""
import os
import time
import pickle as pkl

from transformers import HubertModel, AutoFeatureExtractor
import torch
from torch.utils.data import DataLoader

from mhubert_model.mHuBERT_dataset import AudioDataset, collate_fn
from utils.common_functions import make_dir

class HubertEmbedder:
    """Extracts mHuBERT embeddings from wav files."""
    def __init__(self, device, hidden_layers = [-1], model_name="utter-project/mHuBERT-147", sampling_rate=16000):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, do_normalize=True)
        self.model = HubertModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.eval()
        print(f"DEVICE: {device}")
        self.device = device
        if device == "cuda":
            self.model = self.model.to("cuda")
        self.sampling_rate = sampling_rate
        self.hidden_layers = hidden_layers

    def get_hubert_vectors(self, speech):
        """extract hubert vectors for speech

        Args:
            speech (list[list[float]]): a batch of speech data, where each entry in the batch is a list of floats representing the speech signal
                for a single audio file.

        Returns:
            list[tensor]: a list of tensors holding the requested embeddings. Each entry in the list is a tensor of shape [batch_size, time, hidden_size].
                Each entry represents the embeddings for a specified layer in the model.
        """
        with torch.no_grad():
            normalised_inputs = self.feature_extractor(speech, sampling_rate=self.sampling_rate,
                                                    padding="longest", return_tensors="pt")
            formatted_inputs = normalised_inputs["input_values"]
            if self.device == "cuda":
                formatted_inputs = formatted_inputs.to("cuda")
            output = self.model(formatted_inputs)  # [batch_size, time, hidden_size]
            del formatted_inputs

            outputs = [0] * len(self.hidden_layers)
            for i, layer in enumerate(self.hidden_layers):
                outputs[i] = output.hidden_states[layer].to("cpu").detach()
            del output

            return outputs

    def embed_speech(self, speech):
        hidden_states = self.get_hubert_vectors(speech)
        return hidden_states

if __name__ == "__main__":
    layers = [7]  # from which mHuBERT layers to extract embeddings
    device = "cuda"
    batch_size = 1  # don't change - resulting IR system is much poorer for batch_size>1
    # this is likely since mHuBERT is not trained to embed padded sequences

    top_level_dir = "data/tamil/"
    top_level_embedding_dir = f"{top_level_dir}/embeddings"
    t1 = time.perf_counter()
    hubert = HubertEmbedder(device, hidden_layers=layers)
    t2 = time.perf_counter()
    print(f"TIME TO LOAD HUBERT: {t2-t1:.2f} s")

    t1 = time.perf_counter()
    for folder in ["validation_data", "training_data"]:
        print(f"Generating embeddings for {folder}")
        audio_directory = f"{top_level_dir}/{folder}"
        dataset = AudioDataset(audio_directory)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

        # save dirs for embeddings
        embedding_dirs = []
        for lay in layers:
            embedding_directory = f"{top_level_embedding_dir}/{folder}/{lay}/raw"
            embedding_dirs.append(embedding_directory)
            make_dir(embedding_directory)

        dataset_length = len(dataset)

        for speech, fnames, idxs, max_length in dataloader:
            print(fnames)

            hidden_states = hubert.embed_speech(speech)

            percentage = (idxs[-1] + 1)/dataset_length * 100
            print(f"{percentage:.2f}% done")
            t = time.perf_counter()
            print(f"Time: {t - t1:.2f} s")

            print("SAVING BATCH")
            for index, file in enumerate(fnames):
                for lay_idx, lay in enumerate(layers):
                    embedding_directory = embedding_dirs[lay_idx]
                    embedding_fname = os.path.join(embedding_directory, 
                                                    file.replace(".wav", ".pkl"))
                    with open(embedding_fname, "wb") as f:
                        pkl.dump(hidden_states[lay_idx][index], f)

        t2 = time.perf_counter()
        print(f"TIME TO EXTRACT EMBEDDINGS: {t2-t1:.2f} s")
        print("RUN FINISHED")