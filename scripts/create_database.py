import implementations
import os

data_path = os.path.join(os.path.dirname(os.path.abspath("")),"data")
patched_data_path = os.path.join(os.path.dirname(os.path.abspath("")),"patched_data_ordered")
if not os.path.exists(patched_data_path):
    os.mkdir(patched_data_path)
patch_dimensions = [512,512]
n_patches = 50
implementations.create_database(data_path, patched_data_path, patch_dimensions, n_patches,
                               inference = True, ovl = 0, save_all = False)
