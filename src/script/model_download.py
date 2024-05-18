import os
from huggingface_hub import snapshot_download


# set environment param
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

model_id = "facebook/opt-6.7b"
cache_dir = None
ignore_patterns = ["*.safetensors", "*.msgpack", "*.h5"]
allow_patterns = ['*.json', '*.bin', '*.txt', '*.md']


snapshot_download(
    model_id, cache_dir=cache_dir, ignore_patterns=ignore_patterns, allow_patterns=allow_patterns
)