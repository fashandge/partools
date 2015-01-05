import string
import random

''' This module maintain named references to variables that needs to be shared globally,
especially during multiprocessing to avoid unncessary of read-only large data structure.

usage
-----
use set_global_arg() and remove_global_arg() in pair
>>> import global_manager as gm
>>> global_arg_name = gm.set_global_arg(global_arg)
>>> arg = gm.get_global_arg(global_arg_name)
>>> #use arg to do something
# remember to release the reference to the global variable after use
>>> gm.remove_global_arg(global_arg_name) 
'''

_global_args = {}
_charset = string.ascii_letters + string.digits

def set_global_arg(global_arg):
    name = _generate_global_arg_name()
    global _global_args
    _global_args[name] = global_arg
    return name

def get_global_arg(name):
    return _global_args.get(name, None)

def globals():
    return _global_args.keys()

def remove_global_arg(global_arg_name):
    global _global_args
    if global_arg_name is not None and global_arg_name in _global_args:
        del _global_args[global_arg_name]

def _generate_global_arg_name():
    while True:
        global_arg_name = _random_string(10, prefix='_tmp_global_arg')
        if global_arg_name not in _global_args:
            return global_arg_name

def _random_string(length, prefix='', suffix=''):
    return '{}{}{}'.format(
        prefix+'_' if prefix else '',
        ''.join(random.sample(_charset, length)),
        '_'+suffix if suffix else ''
    )

