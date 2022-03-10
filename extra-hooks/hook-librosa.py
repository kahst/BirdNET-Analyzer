from PyInstaller.utils.hooks import collect_data_files
datas = collect_data_files('librosa')
hiddenimports=['scipy._lib.messagestream', 'sklearn.tree', 'sklearn.neighbors.typedefs','sklearn.neighbors._partition_nodes', 'sklearn.neighbors.quad_tree', 'sklearn.tree._utils', 'sklearn.utils._typedefs']