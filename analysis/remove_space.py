import shutil
import os


def remove_space(file_dir):
    for file_name in os.listdir(file_dir):
        new_file_name = file_name.replace(" ", "_")
        shutil.move(os.path.join(file_dir, file_name), os.path.join(file_dir, new_file_name))


remove_space("/mnt/data/feature_extraction/movie/selected_frames/images_clahe_crop")
