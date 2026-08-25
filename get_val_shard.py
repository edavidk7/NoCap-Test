import os
from huggingface_hub import hf_hub_download

local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fineweb10B")
os.makedirs(local_dir, exist_ok=True)

path = hf_hub_download(
    repo_id="kjj0/fineweb10B-gpt2",
    filename="fineweb_val_000000.bin",
    repo_type="dataset",
    local_dir=local_dir,
)
print("downloaded to", path)
