import os
import torch
import esm
import gzip
import pickle
import h5py
import numpy as np

# Set custom cache directory for ESM model
os.environ["TORCH_HOME"] = "/data/khaniyau2/deep_learning/esm2/esm_models"

def load_data(file_path):
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_embeddings_hdf5(embeddings, output_file):
    print(f"[INFO] Saving all embeddings to HDF5: {output_file}")
    with h5py.File(output_file, "w") as h5f:
        for protein_id, residue_dict in embeddings.items():
            group = h5f.create_group(protein_id)
            for res_id, emb_tensor in residue_dict.items():
                emb_array = emb_tensor.cpu().numpy() if isinstance(emb_tensor, torch.Tensor) else np.asarray(emb_tensor)
                group.create_dataset(res_id, data=emb_array, compression="gzip", compression_opts=4)
    print(f"[INFO] Finished writing to {output_file}")

def get_esmfold_embeddings(data, model_name):
    print(f"[INFO] Loading model: {model_name}")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[INFO] Using device: {device}")

    batch_converter = alphabet.get_batch_converter()
    result = {}

    for key, residues in data.items():
        sequence = ''.join(res[0] for res in residues)
        residue_ids = [f"{res[1]}_{res[0]}" for res in residues]

        batch = [(key, sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.to(device)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            output = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=False)

        token_representations = output["representations"][model.num_layers]
        embeddings = token_representations[0, 1:batch_lens[0] - 1].cpu()

        if len(embeddings) != len(residue_ids):
            raise ValueError(f"Embedding/residue mismatch for {key}")

        result[key] = {res_id: emb for res_id, emb in zip(residue_ids, embeddings)}

    return result

if __name__ == "__main__":
    MODEL_NAME = "esm2_t33_650M_UR50D"
    INPUT_FILE = "/data/khaniyau2/deep_learning/esm2/esm2_t33_650M_UR50D_embedding/sequences_input_train_test.pkl.gz"
    OUTPUT_FILE = f"{MODEL_NAME}_all_embeddings.h5"

    print("[INFO] Loading input data...")
    input_data = load_data(INPUT_FILE)
    print(f"[INFO] Loaded {len(input_data)} protein entries")

    print("[INFO] Generating ESM embeddings...")
    all_embeddings = get_esmfold_embeddings(input_data, MODEL_NAME)

    print("[INFO] Saving to HDF5...")
    save_embeddings_hdf5(all_embeddings, OUTPUT_FILE)

