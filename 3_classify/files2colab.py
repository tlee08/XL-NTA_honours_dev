import os
import shutil
from glob import glob

# From and To root folders
from_dir = "/Volumes/tim_details/tim_honours/CAPTURES/"
to_dir = "/Users/timothylee/Desktop/Captures"

# For each {device}_{location}
for i in os.listdir(from_dir):
    print(i)
    # For each {content}
    for j in os.listdir(os.path.join(from_dir, i)):
        print(j)
        # Copying wlan h5 files to corresponding directory
        shutil.copytree(
            os.path.join(from_dir, i, j, "wlan_h5"),
            os.path.join(to_dir, i, j, "wlan_h5"),
        )
        # Copying csi h5 files to corresponding directory
        shutil.copytree(
            os.path.join(from_dir, i, j, "csi_h5"),
            os.path.join(to_dir, i, j, "csi_h5"),
        )
    print()
