import os
import zipfile
from pathlib import Path
import shutil

# Define input root and output folder
input_root = Path("/home/s27mhusa_hpc/Master-Thesis/admin-510084584112192272708/annotation")
output_dir = Path("/home/s27mhusa_hpc/Master-Thesis/XMI_Files_OpenAgrar")
output_dir.mkdir(parents=True, exist_ok=True)

# Traverse folders in the annotation directory
for folder in input_root.iterdir():
    if folder.is_dir():
        zip_path = folder / "GolzL.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # List all files in the zip
                for file_name in zip_ref.namelist():
                    if file_name.endswith("GolzL.xmi"):
                        # Construct new file name (strip '_inception' from folder name)
                        new_name = folder.name.replace("_inception", "")
                        dest_path = output_dir / new_name

                        # Extract GolzL.xmi to a temp location and move it
                        with zip_ref.open(file_name) as source, open(dest_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
                        print(f"Extracted and renamed: {dest_path}")
