from .parmap import map, groupby_apply

__version__ = '0.1.1'

def _readme():
    from os import path
    from codecs import open
    here = path.dirname(path.abspath(path.dirname(__file__)))
    with open(path.join(here, 'README.md'), encoding='utf-8') as f:
        return f.read()

__doc__ = _readme()
