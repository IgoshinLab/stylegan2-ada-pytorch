import tifffile
import os


def get_frames(tif_file, out_dir, step=30, end_frame=None):
    img_name = os.path.basename(tif_file).split('.')[0]
    img_stack = tifffile.TiffFile(tif_file).asarray()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not end_frame:
        end_frame = len(img_stack)
    for i in range(0, end_frame, step):
        img = img_stack[i, :, :]
        tifffile.imwrite(os.path.join(out_dir, "%s_%i.tif" % (img_name, i)), img)


get_frames("/mnt/data/feature_extraction/movie/tifs/AG1110_0/AG1110 071817_867.tif",
           "/mnt/data/feature_extraction/movie/selected_frames/images", step=60, end_frame=None)