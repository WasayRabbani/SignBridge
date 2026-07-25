import os
import re


def rename_videos(folder_path, action_name):
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    
    # Get all video files
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    
    # Find the highest existing number
    existing_max = 0
    files_to_rename = []
    
    for f in files:
        # Check if it already matches "Word_X.mp4"
        match = re.match(rf"{re.escape(action_name)}_(\d+)\.\w+$", f, re.IGNORECASE)
        if match:
            existing_max = max(existing_max, int(match.group(1)))
        else:
            files_to_rename.append(f)
            
    if not files_to_rename:
        return

    print(f"[{action_name}] Found {len(files_to_rename)} new files to rename.")

    # Rename the remaining files starting from max + 1
    for f in files_to_rename:
        existing_max += 1
        extension = os.path.splitext(f)[1]
        new_name = f"{action_name}_{existing_max}{extension}"
        source = os.path.join(folder_path, f)
        destination = os.path.join(folder_path, new_name)
        
        os.rename(source, destination)
        print(f"  ✅ {f} → {new_name}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    INPUT_FOLDER = r"D:\Signs"
    if os.path.exists(INPUT_FOLDER):
        print(f"Scanning {INPUT_FOLDER} for new videos to rename...\n")
        word_folders = [f for f in os.listdir(INPUT_FOLDER) if os.path.isdir(os.path.join(INPUT_FOLDER, f))]
        
        for word in word_folders:
            folder_path = os.path.join(INPUT_FOLDER, word)
            rename_videos(folder_path, word)
        
        print("\nRenaming complete.")
    else:
        print(f"Warning: {INPUT_FOLDER} not found.")