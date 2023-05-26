import os
import platform

from multiprocessing import set_start_method

data_path = os.path.join(os.path.dirname(__file__), "test_data")

if platform.system() == 'Windows' or platform.system() == "Darwin":
    set_start_method("spawn")
else:
    set_start_method("forkserver")


def get_test_data(filename):
    return os.path.join(data_path, filename)
