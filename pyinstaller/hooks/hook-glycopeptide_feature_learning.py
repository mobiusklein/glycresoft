from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = collect_submodules(
    "glycopeptide_feature_learning._c") + collect_submodules("glycopeptide_feature_learning.scoring._c")

datas = collect_data_files('glycopeptide_feature_learning')
