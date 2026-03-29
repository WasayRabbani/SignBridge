import os

def rename_videos(folder_path, action_name):
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    
    # Get all video files
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    
    # Sort by modification time — preserves recording order regardless of filename
    files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))

    print(f"Found {len(files)} files in: {folder_path}")
    print(f"Renaming to: {action_name}_1, {action_name}_2, ...\n")

    for index, filename in enumerate(files):
        extension = os.path.splitext(filename)[1]
        new_name = f"{action_name}_{index + 1}{extension}"
        
        source      = os.path.join(folder_path, filename)
        destination = os.path.join(folder_path, new_name)

        # Skip if already correctly named
        if filename == new_name:
            print(f"  ✅ Already correct: {filename}")
            continue
        
        # Safety check — don't overwrite existing files
        if os.path.exists(destination):
            print(f"  ⚠️  SKIPPED {filename} — {new_name} already exists")
            continue

        os.rename(source, destination)
        print(f"  ✅ {filename} → {new_name}")

    print(f"\nDone. {len(files)} files renamed.")

# ============================================================
# CHANGE THESE TWO LINES ONLY
# ============================================================
rename_videos(r"D:\Signs\Food", 'Food')