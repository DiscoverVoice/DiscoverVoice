import os
import subprocess

# Directory for cutting mp3 files
cur_dir = os.path.dirname(os.path.realpath(__file__))
directory = os.path.join(cur_dir, "../Datasets")

for filename in os.listdir(directory):
    if filename.endswith(".mp3"):
        # Get a source file path
        original_path = os.path.join(directory, filename)
        # Set a file name
        new_filename = filename.replace(".mp3", "_cut.mp3")
        new_path = os.path.join(directory, new_filename)
        command = f"ffmpeg -i {original_path} -ss 00:00:40 -to 00:01:40 -c copy {new_path}"
        subprocess.run(command, shell=True)

print("All files have been cut successfully")