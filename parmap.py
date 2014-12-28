#   Copyright 2014 Jianfu Chen 
#   csjfchen *AT* gmail
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

def map(func, local_args, global_arg=None,
        chunksize=1, processes=2):
    '''
    global_arg: a global variable shared with all processes intended to be read-only
    so that we can utilize the copy-on-write mechanism in linux to avoid unnecceary copy of data,
    this is good for parallel processing of large data structure, like a big pandas dataframe (pass
    df as global_arg to avoid unncessary copy, and indices of subset as local_args that is to be copied)
    '''
    global_arg_name = None
    try:
        if global_arg is not None:
            global_arg_name = random_string(10, prefix='_tmp_global_arg')
            globals()[global_arg_name] = global_arg
            def func_with_global(local_args, global_arg_name):
                global_arg = globals()[global_arg_name] 
                return func(local_args, global_arg)
                
            process_func = toolz.partial(
                func_with_global,
                global_arg_name=global_arg_name)
        else:
            process_func = func

        if processes == 1:
            result = list(toolz.map(process_func, local_args))
        else:
            import pathos.multiprocessing as mp
            pool = mp.Pool(processes=processes)
            result = pool.map(process_func, local_args, chunksize=chunksize)
            pool.close()
            pool.join()

        return result
    except Exception as ex:
        print 'exception: {}'.format(ex)
    finally:
        if global_arg_name is not None:
            del globals()[global_arg_name]

def random_string(length, prefix='', suffix=''):
    import string
    return '{}{}{}'.format(
        prefix+'_' if prefix else '',
        ''.join(random.sample(string.ascii_letters + string.digits, length)),
        '_'+suffix if suffix else ''
    )
