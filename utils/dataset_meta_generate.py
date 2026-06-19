import os
import csv
import json

# --- CONFIGURATION ---
# Path where your organized dataset folders ("0010", "0011", etc.) are located
DATASET_DIR = r"D:\soft\LGraph2\dataset_organized"

# Path to the CSV file you saved in step 1
CSV_FILE_PATH = "utils/labels.csv"

# Fixed parameters as specified for this dataset:
SAMPLE_RATE_CURRENT_HZ = 50000.0
SAMPLE_RATE_VIBRO_HZ = 29296.0
PWM_FREQUENCY_HZ = 4000
# ---------------------

def get_class_name(displacement: float) -> str:
    """
    Translates displacement floats to class names.
    Examples:
      0.0   -> "d0"
      0.32  -> "d0_32"
      -0.12 -> "d_neg_0_12"
    """
    if displacement == 0.0:
        return "d0"
    
    prefix = "d_neg_" if displacement < 0 else "d"
    # Format absolute value to string and swap the dot for an underscore
    disp_str = str(abs(displacement)).replace('.', '_')
    return f"{prefix}{disp_str}"

def generate_meta_files():
    if not os.path.exists(CSV_FILE_PATH):
        print(f"Error: CSV file '{CSV_FILE_PATH}' not found.")
        return

    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found.")
        return

    generated_count = 0

    with open(CSV_FILE_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # 1. Format the folder name to be exactly 4 digits (e.g. "10" -> "0010")
            raw_folder = row['folder_name'].strip()
            folder_name = raw_folder.zfill(4)
            
            try:
                displacement = float(row['vertical_displacement_mm'])
                frequency = float(row['electrical_frequency_hz'])
                load_level = int(row['load_level'])
            except ValueError as e:
                print(f"Skipping row for folder {folder_name} due to conversion error: {e}")
                continue

            # 2. Get the standardized class name (e.g. "d0_32" or "d0")
            class_name = get_class_name(displacement)

            # 3. Locate the destination folder
            folder_path = os.path.join(DATASET_DIR, folder_name)
            
            if not os.path.exists(folder_path):
                # Prints a warning if a folder on the list is missing in your actual directory
                print(f"Warning: Folder '{folder_path}' does not exist. Skipping.")
                continue

            # 4. Construct JSON payload
            meta_content = {
                "class": class_name,
                "electrical_frequency_hz": frequency,
                "load": load_level,
                "sample_rate_current_hz": SAMPLE_RATE_CURRENT_HZ,
                "sample_rate_vibro_hz": SAMPLE_RATE_VIBRO_HZ,
                "pwm_frequency_hz": PWM_FREQUENCY_HZ
            }

            # 5. Write the meta.json file
            json_path = os.path.join(folder_path, "meta.json")
            with open(json_path, 'w', encoding='utf-8') as json_file:
                json.dump(meta_content, json_file, indent=2, ensure_ascii=False)
            
            generated_count += 1

    print(f"\nProcessing complete. Generated meta.json for {generated_count} folders.")

if __name__ == "__main__":
    generate_meta_files()