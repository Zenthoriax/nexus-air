import os
from huggingface_hub import hf_hub_download
from config import settings

def download_model():
    model_dir = settings.llm_models_path
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Downloading Phi-3-mini Q4_K_M GGUF to {model_dir}...")
    
    # We will use the canonical Microsoft repo for the GGUF weights
    repo_id = "microsoft/Phi-3-mini-4k-instruct-gguf"
    filename = "Phi-3-mini-4k-instruct-q4.gguf"
    
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=model_dir,
        local_dir_use_symlinks=False # Important so we get the actual file, not a symlink
    )
    
    print(f"Successfully downloaded to: {file_path}")
    print(f"File size: {os.path.getsize(file_path) / (1024 * 1024 * 1024):.2f} GB")

if __name__ == "__main__":
    download_model()
