import zipfile
from pathlib import Path
import shutil

# Define input root and output folder
input_root = Path("/home/s27mhusa_hpc/Master-Thesis/curation")
output_dir = Path("/home/s27mhusa_hpc/Master-Thesis/NewDatasets27August/XMI_Files_Bonares")
output_dir.mkdir(parents=True, exist_ok=True)

for folder in input_root.iterdir():
    if folder.is_dir():
        # Look for any .zip file in the folder
        for zip_path in folder.glob("*.zip"):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # List all files in the zip
                for file_name in zip_ref.namelist():
                    if file_name.endswith("CURATION_USER.xmi"):
                        # Construct new file name (strip '_inception' from folder name)
                        new_name = folder.name.replace("_inception", "")
                        dest_path = output_dir / (new_name)

                        # Extract and copy the file
                        with zip_ref.open(file_name) as source, open(dest_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
                        print(f"Extracted and renamed: {dest_path}")
