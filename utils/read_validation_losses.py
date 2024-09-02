"""Read validation losses from models saved in a directory."""
import os

from awe_model.train_model import load_model
from awe_model.model import SSEmodel

if __name__ == "__main__":
    language = "tamil"
    layer = 9
    min_phone_seq_length = 3
    max_phone_seq_length = 9
    device = "cpu"
    model_save_dir = \
        f"data/{language}/models/{layer}/{min_phone_seq_length}_{max_phone_seq_length}"

    
    model = SSEmodel(device=device)

    model_files = sorted(os.listdir(model_save_dir))

    for model_file in model_files:
        state_dict = load_model(f"{model_save_dir}/{model_file}", model, device)
        valid_loss = state_dict["validation_loss"]
        print(f"Validation loss: {valid_loss}")
