import os
import pickle
import requests
import numpy as np
import tiktoken

input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

# Replace the URL with a link to a larger Shakespeare dataset
data_url = 'https://raw.githubusercontent.com/dscape/spell/master/test/resources/big.txt'
response = requests.get(data_url)
if response.status_code == 200:
    with open(input_file_path, 'w') as f:
        f.write(response.text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# initialize the tiktoken encoding for OpenAI models
encoder = tiktoken.get_encoding("gpt2")

# encode the data using tiktoken
train_data = data[:int(len(data) * 0.9)]
val_data = data[int(len(data) * 0.9):]

train_ids = encoder.encode(train_data)
val_ids = encoder.encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
print(f"size { len(set(train_ids))}, {len(set(val_ids)-set(train_ids))} {len(set(val_ids))}")
# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well
meta = {
    'encoder_name': 'gpt2',
    'vocab_size': encoder.n_vocab,
    'special_tokens': encoder._special_tokens,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
