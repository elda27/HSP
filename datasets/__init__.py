from .dataset import Dataset


def get_crossval_patients(xls_file, index):
    import pandas as pd
    from distutils.version import LooseVersion
    if LooseVersion(pd.__version__) >= LooseVersion('0.21.0'):
        df = pd.read_excel(xls_file, sheet_name=index)
    else:
        df = pd.read_excel(xls_file, sheetname=index)

    train = df['train'].dropna().tolist()
    valid = df['valid'].dropna().tolist()
    test = df['test'].dropna().tolist()
    return {'train': train, 'valid': valid, 'test': test}
