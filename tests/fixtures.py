import os

data_path = os.path.join(os.path.dirname(__file__), "test_data")


def get_test_data(filename):
    return os.path.join(data_path, filename)
