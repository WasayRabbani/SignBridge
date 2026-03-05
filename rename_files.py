import os

# Claude
def rename_videos(folder_path, action_name):
    # Supported video formats
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    
    # Get and sort files to maintain order
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    files.sort(key=lambda f: int(os.path.splitext(f)[0].split('_')[-1]) if os.path.splitext(f)[0].split('_')[-1].isdigit() else 0) 

    print(f"Renaming {len(files)} files in: {folder_path}")

    for index, filename in enumerate(files):
        extension = os.path.splitext(filename)[1]
        # New name format: ActionName_Number.extension
        new_name = f"{action_name}_{index + 1}{extension}"
        
        source = os.path.join(folder_path, filename)
        destination = os.path.join(folder_path, new_name)
        
        os.rename(source, destination)
        print(f"Done: {filename} -> {new_name}")

# --- HOW TO USE ---
# 1. Change 'C:/MyVideos/I' to your actual folder path
# 2. Change 'I' to the word in that folder
rename_videos(r"D:\Signs\Water", 'Water')
