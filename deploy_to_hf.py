import os
from huggingface_hub import HfApi, login

print("To deploy to Hugging Face, you need an Access Token with 'Write' permissions.")
print("Get one here: https://huggingface.co/settings/tokens")
token = input("Enter your Hugging Face Access Token: ").strip()

login(token=token)
api = HfApi()

repo_id = "benzmehdi/nlppro"
print(f"\nUploading files to {repo_id}...")

# List of folders/files to upload
files_to_upload = [
    "backend",          # includes backend/requirements.txt
    "module 1 models",
    "summ_artifacts",
    "module 3 model",
    "module 4 distilbert",
    "Dockerfile",
    "glove.6B.100d.txt"
]

for item in files_to_upload:
    if not os.path.exists(item):
        print(f"Skipping {item} (not found)")
        continue
        
    if os.path.isfile(item):
        print(f"Uploading file: {item}")
        api.upload_file(
            path_or_fileobj=item,
            path_in_repo=item,
            repo_id=repo_id,
            repo_type="space"
        )
    else:
        print(f"Uploading folder: {item}")
        api.upload_folder(
            folder_path=item,
            path_in_repo=item,
            repo_id=repo_id,
            repo_type="space"
        )

print("\n✅ Upload complete! Your Space is now building.")
print(f"Check the status at: https://huggingface.co/spaces/{repo_id}")
