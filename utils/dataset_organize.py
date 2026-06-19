# How to use this script:

# 1.  Make sure you have Python installed on your system.
# 2.  Save this script as a Python file (e.g., organize.py).
# 3.  If necessary, adjust the SOURCE_DIR and OUTPUT_DIR paths in the
#     configuration section of the script.
# 4.  Run the script from your terminal or command prompt:
#     python organize.py

# Details on how it works:

#  - re.compile(r"^test_(\d{4})_.*\.dat$"): This searches specifically for
#    filenames starting with test_ followed by exactly 4 digits, followed by any
#    characters, and ending in .dat. It ignores other files like .gfl, .m, or
#    .par.
#  - shutil.copy2: This is used instead of standard copy to preserve the original
#    file metadata (such as file creation and modification timestamps).


import os
import re
import shutil

# --- CONFIGURATION ---
# Path to the directory containing your source files
SOURCE_DIR = r"D:\soft\LGraph2\dataset"

# Path where you want the organized dataset to be created
# (It is recommended to output to a separate folder to keep the original files intact)
OUTPUT_DIR = r"D:\soft\LGraph2\dataset_organized"
# ---------------------

# Regular expression to match "test_XXXX_..." .dat files and capture the 4-digit number (XXXX)
file_pattern = re.compile(r"^test_(\d{4})_.*\.dat$", re.IGNORECASE)

def organize_dataset():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: The source directory '{SOURCE_DIR}' does not exist.")
        return

    # Create the output root directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    copied_count = 0

    # Iterate through all files in the source directory
    for filename in os.listdir(SOURCE_DIR):
        src_file_path = os.path.join(SOURCE_DIR, filename)
        
        # Ensure we are only processing files (not subfolders)
        if os.path.isfile(src_file_path):
            match = file_pattern.match(filename)
            if match:
                # Extract the 4-digit folder name (e.g., "0002")
                folder_id = match.group(1)
                
                # Define the destination folder path
                dest_folder_path = os.path.join(OUTPUT_DIR, folder_id)
                os.makedirs(dest_folder_path, exist_ok=True)
                
                # Copy the file to the new folder
                dest_file_path = os.path.join(dest_folder_path, filename)
                shutil.copy2(src_file_path, dest_file_path)
                
                print(f"Copied: {filename} -> {folder_id}/")
                copied_count += 1

    print(f"\nCompleted. Copied {copied_count} files to '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    organize_dataset()


