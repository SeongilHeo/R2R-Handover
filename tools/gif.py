import imageio
import os

folder_path = "frames"
files = os.listdir(folder_path)
files.sort()

with imageio.get_writer("robot_motion.gif", mode='I', duration=0.1) as writer:
    for fname in files:
        try:
            fname = os.path.join(folder_path, fname)
            image = imageio.imread(fname)
            writer.append_data(image)
        except FileNotFoundError:
            break

# Delete
for fname in files:
    os.remove(fname)