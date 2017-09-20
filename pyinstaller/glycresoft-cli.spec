# -*- mode: python -*-

block_cipher = None


a = Analysis(['glycresoft-cli.py'],
             pathex=['C:\\Users\\Owner\\Dev\\glycresoft\\pyinstaller'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=['./'],
             runtime_hooks=[],
             excludes=['_tkinter', 'PyQt4', 'PyQt5', 'IPython', 'pandas'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='glycresoft-cli',
          debug=False,
          strip=False,
          upx=True,
          console=True , icon='img\\logo.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='glycresoft-cli')
