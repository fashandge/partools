import string
import random

_global_args = {}
_charset = string.ascii_letters + string.digits

def get_global_arg(name):
    return _global_args.get(name, None)

def globals():
    return _global_args.keys()

def remove_global_arg(global_arg_name):
    global _global_args
    if global_arg_name is not None and global_arg_name in _global_args:
        del _global_args[global_arg_name]

def set_global_arg(global_arg):
    name = _generate_global_arg_name()
    global _global_args
    _global_args[name] = global_arg
    return name

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

