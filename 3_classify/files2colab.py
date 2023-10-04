import os
import shutil
from glob import glob

# Going through each of the <device>_<location> folders
from_dir = "/Volumes/tim_details/tim_honours/CAPTURES/"
to_dir = "/Users/timothylee/Desktop/Captures"


for i in os.listdir(from_dir):
    print(i)
    from_dir_i = os.path.join(from_dir, i)
    for j in os.listdir(from_dir_i):
        print(j)
        # Making directory
        os.makedirs(os.path.join(to_dir, i, j), exist_ok=True)
        # Copying csi_all.npy to corresponding directory
        shutil.copyfile(
            os.path.join(from_dir, i, j, "csi_all.npy"),
            os.path.join(to_dir, i, j, "csi_all.npy")
        )
    print()
