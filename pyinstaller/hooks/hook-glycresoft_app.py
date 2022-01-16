
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

datas = list(filter(lambda x: "test_data" not in x[1], collect_data_files("glycresoft_app")))
