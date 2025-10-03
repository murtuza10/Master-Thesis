import os
import shutil
from pathlib import Path

def move_files_simple(file_name_list, source_folder, destination_folder, auto_overwrite=False):
    """
    Simplified version without user prompts (good for scripts)
    
    Args:
        file_name_list (list): List of file names to move
        source_folder (str): Path to source folder
        destination_folder (str): Path to destination folder
        auto_overwrite (bool): If True, automatically overwrite existing files
    """
    
    # Create destination folder if it doesn't exist
    Path(destination_folder).mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    failed_count = 0
    
    for file_name in file_name_list:
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        
        try:
            if os.path.exists(source_path):
                if os.path.exists(destination_path) and not auto_overwrite:
                    print(f"⚠️  Skipped (already exists): {file_name}")
                    failed_count += 1
                    continue
                    
                shutil.move(source_path, destination_path)
                print(f"✅ Moved: {file_name}")
                moved_count += 1
            else:
                print(f"❌ Not found: {file_name}")
                failed_count += 1
                
        except Exception as e:
            print(f"❌ Error moving {file_name}: {str(e)}")
            failed_count += 1
    
    print(f"\nSummary: {moved_count} moved, {failed_count} failed")

# Example usage
if __name__ == "__main__":
    # Example file list
    file_name = [
"95393.xmi",
"96125.xmi",
"99969.xmi",
"98225.xmi",
"96100.xmi",
"97508.xmi",
"97504.xmi",
"95392.xmi",
"95220.xmi",
"95965.xmi",
"97516.xmi",
"96249.xmi",
"95964.xmi",
"96271.xmi",
"97456.xmi",
"97499.xmi",
"97524.xmi",
"97437.xmi",
"96247.xmi",
"97455.xmi",
"96273.xmi",
"97452.xmi",
"96270.xmi",
"96079.xmi",
"97532.xmi",
"96191.xmi",
"96255.xmi",
"97506.xmi",
"97454.xmi",
"96088.xmi",
"97527.xmi",
"97494.xmi",
"97450.xmi",
"97514.xmi",
"97435.xmi",
"97457.xmi",
"97099.xmi",
"98874.xmi",
"96138.xmi",
"96080.xmi",
"97447.xmi",
"95687.xmi",
"95684.xmi",
"95233.xmi",
"97495.xmi",
"98829.xmi",
"96246.xmi",
"97496.xmi",
"95232.xmi",
"96086.xmi",
"97449.xmi",
"97111.xmi",
"97446.xmi",
"97430.xmi",
"96078.xmi",
"97498.xmi",
"97100.xmi",
"97431.xmi",
"97109.xmi",
"98842.xmi",
"97464.xmi",
"99959.xmi",
"95391.xmi",
"97511.xmi",
"96272.xmi",
"95221.xmi",
"97517.xmi",
"97444.xmi",
"97513.xmi",
"96245.xmi",
"95683.xmi",
"96061.xmi",
"97101.xmi",
"98269.xmi",
"96254.xmi",
"97426.xmi",
"97458.xmi",
"102202.xmi",
"100745.xmi",
"76443.xmi",
"92844.xmi",
"90310.xmi",
"94591.xmi",
"94595.xmi",
"94386.xmi",
"94605.xmi",
"76474.xmi",
"94367.xmi",
"94212.xmi",
"92711.xmi",
"94593.xmi",
"94364.xmi",
"41962.xmi",
"76473.xmi",
"94589.xmi",
"101687.xmi",
"100507.xmi",
"93065.xmi",
"94368.xmi",
"95217.xmi",
"95216.xmi",
"42380.xmi",
"58736.xmi",
"94389.xmi",
"100752.xmi",
"94397.xmi",
"76441.xmi",
"94510.xmi",
"94405.xmi",
"93221.xmi",
"84145.xmi",
"100753.xmi",
"101681.xmi",
"94520.xmi",
"87123.xmi",
"65536.xmi",
"94606.xmi",
"54877.xmi",
"76475.xmi",
"76442.xmi",
"94514.xmi",
"94590.xmi",
"94603.xmi",
"94513.xmi",
"41959.xmi",
"76438.xmi",
"94567.xmi",
"76471.xmi",
"94515.xmi",
"76472.xmi",
"62742.xmi",
"75383.xmi",
"94587.xmi",
"100506.xmi",
"94565.xmi",
"76440.xmi",
"94370.xmi",
"100749.xmi",
"94393.xmi"]

    
    # Example paths
    source_folder = "/home/s27mhusa_hpc/Master-Thesis/Dataset19September/Test_XMI_Files_English"
    destination_folder = "/home/s27mhusa_hpc/Master-Thesis/Dataset19September/Test_XMI_Files_German"


    # Method 2: Simple mode without prompts
    move_files_simple(file_name, source_folder, destination_folder, auto_overwrite=True)
    
    print("Replace the example paths and file list with your actual values!")


