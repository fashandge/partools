#! /usr/bin/env python
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
import sys
import cPickle
import random
import string
import toolz
try:
    import pathos.multiprocessing as mp_pathos
except:
    mp_pathos = None
import multiprocessing as mp_std
try:
    # import modules related for parallel processing methods 
    # for pandas dataframe
    import pandas as pd
    import numpy as np
    #from IPython.core.debugger import Tracer
    #import src.utils.utilities as util
except:
    pass

# Used as the timeout
GOOGLE = 1e100

def map(func, iterable, global_arg=None,
        chunksize=1, processes=2, use_pathos=False):
    ''' A parallel version of standard map function. It is a blocking operation and return ordered results.

    If you want to process a read-only large data structure by parts in parallel, put it in the global_arg. 
    This function helps to avoid unncessary copy of that large data to be memory and computaton efficient.
    This is enabled by the copy-and-write feature of fork in Linux (but not in windows).

    Parameters
    ----------
    func : worker function func with one or two arguments
        1. global_arg is None: func(local_arg) should accept one argument
        2. global_arg is not None: func(local_arg, global_arg) should accept two arguments
        To use standard multiprocessing module (use_pathos=False), this function usually is a function 
        in a module or a partial function (say using toolz or functools). For inner function or lambda functions,
        need to use_pathos for correct pickling.

    iterable : iterable
        The sequence of data you want to process. They will be copied to children processes. 
        So usually set it to be small data structue, e.g., the array indices. 

    global_arg : object
        the large object you want to share with children processes, but don't want to copy 
        to children process, like a big pandas dataframe, or a large numpy array. The worker function should use 
        it as if it is read-only, otherwise expensive copy operation follows. 

    chunksize : int
        number of items to be assigned to a child process at a time. Same as in the standard Pool.map.

    processes : int 
        number of children processes (workers) in the pool, set it up to the number of physical cores.
        Same as in the standard Pool.map. If processes=1, it is equivalent to non-parallel map.

    use_pathos: boolean, default False
        use pathos.multiprocessing or standard multiprocessing package. Recommend to use latter most of the time
        as it is faster in general. Only when your function cannot be pickled will you consider use pathos.

    Returns
    -------
    a list of objects each of which is the processing result of an element in the iterable iterable

    Raises
    ------
    KeyboardInterrupt, Exception

    Examples
    --------
    See main function.
    '''
    if not use_pathos and processes != 1:
        _trypickle(func)
    global_arg_name = None
    pool = None
    try:
        if global_arg is None:
            process_func = func
        else:
            global_arg_name = _random_string(10, prefix='_tmp_global_arg')
            # use a temporary global variable to hold the large object global_arg
            globals()[global_arg_name] = global_arg

            # partial function can be pickled by multiprocessing
            # but not inner function
            process_func = toolz.partial(
                _func_with_global,
                global_arg_name=global_arg_name,
                func=func
            )

        if processes == 1:
            import __builtin__
            result = __builtin__.map(process_func, iterable)
        else:
            if use_pathos and mp_pathos is None:
                raise Exception()
            mp = mp_pathos if use_pathos else mp_std
            pool = mp.Pool(processes=processes)
            result = pool.map_async(process_func, 
                                    iterable, chunksize=chunksize).get(GOOGLE)

        return result
    except KeyboardInterrupt:
        print 'keyboard interrupt'
        raise KeyboardInterrupt
    except Exception as ex:
        print 'exception in parallel map: {}'.format(ex)
        raise ex
    finally:
        glb = globals()
        if global_arg_name is not None and global_arg_name in glb:
            del glb[global_arg_name]
        _terminate_pool(pool)

def _terminate_pool(pool):
    if pool is not None:
        pool.close()
        pool.terminate()
        pool.join()

# wrap the original worker function using a temporary global variable name
# so that we just pass a name (string) instead of the actual big object (global_arg) to 
# the wrapped worker function
def _func_with_global(local_arg, global_arg_name, func):
    try:
        global_arg = globals()[global_arg_name] 
        return func(local_arg, global_arg)
    except KeyboardInterrupt:
        print 'KeyboardInterrupt'

_charset = string.ascii_letters + string.digits
def _random_string(length, prefix='', suffix=''):
    return '{}{}{}'.format(
        prefix+'_' if prefix else '',
        ''.join(random.sample(_charset, length)),
        '_'+suffix if suffix else ''
    )

def _trypickle(func):
    """
    Attempts to pickle func since multiprocessing needs to do this.

    copied from Rosseta package.
    """
    genericmsg = "Pickling of func (necessary for multiprocessing) failed."

    boundmethodmsg = genericmsg + '\n\n' + """
    func contained a bound method, and these cannot be pickled.  This causes
    multiprocessing to fail.  Possible causes/solutions:

    Cause 1) You used a lambda function or an object's method, e.g.
        my_object.myfunc
    Solution 1) Wrap the method or lambda function, e.g.
        def func(x):
            return my_object.myfunc(x)

    Cause 2) You are pickling an object that had an attribute equal to a
        method or lambda func, e.g. self.myfunc = self.mymethod.
    Solution 2)  Don't do this.
    """

    try:
        cPickle.dumps(func)
    except TypeError as e:
        if 'instancemethod' in e.message:
            sys.stderr.write(boundmethodmsg + "\n")
        else:
            sys.stderr.write(genericmsg + '\n')
        raise
    except:
        sys.stderr.write(genericmsg + '\n')
        raise

''' ========================
functions for parallel groupby().apply for pandas dataframes
'''

def groupby_apply(df, by, apply_func, 
                  algorithm='default', 
                  processes=2, chunksize=-1,
                  use_pathos=False, 
                  **groupby_kwargs):
    '''Group a pandas dataframe and process groups in parallel, similar to df.groupby().apply

    Parameters
    ----------
    df : pandas.DataFrame
    by : string or a list of string
        column names to groupby
    apply_func : function
        apply function
    algorithm : string, default 'default'
        specify which algorithm to use. Leave it as default most of the time
    chunksize : int, default -1
        use -1 to choose chunksize automatically 
    groupby_kwargs : optional keyword arguments
        optional keyword arguments for df.groupby

    Returns
    -------
    A dataframe : usually the same as non-parallel version
    '''
    # first groupby df, iterate and process each group
    # the grouped object is in global_arg, hence avoid copying
    if algorithm == 'default' or algorithm=='iter':
        alg = _groupby_apply_iter

    # first groupby df, iterate and process each group
    # the grouped object is in local_arg
    elif algorithm == 'iter_local':
        alg = _groupby_apply_iter_local

    # first groupby df, then split df into sections without 
    # splitting a same group; then do groupy.apply for each df section
    elif algorithm == 'split':
        alg = _groupby_apply_split

    return alg(df, by, apply_func,
               processes=processes, chunksize=chunksize,
               use_pathos=use_pathos,
               **groupby_kwargs)

def _groupby_apply_iter_local(df, by, apply_func, processes=2,
                                   chunksize=-1, use_pathos=False,
                                   **groupby_kwargs):
    grouped = df.groupby(by, **groupby_kwargs)
    if chunksize == -1:
        default_chunksize = 10
        chunksize = min(default_chunksize, 
                        _auto_chunksize(grouped.ngroups, processes))
    process_group = toolz.partial(
        _process_named_group, by=by, 
        apply_func=apply_func)
    labels_values = map(process_group, grouped, 
                        processes=processes,
                        chunksize=chunksize,
                        use_pathos=use_pathos)
    result = _combine_apply_results(labels_values, by)    
    return result

def _groupby_apply_iter(df, by, apply_func, processes=2,
                        chunksize=-1, use_pathos=False,
                        **groupby_kwargs):
    grouped = df.groupby(by, **groupby_kwargs)
    n_g = [(name, group) for name, group in grouped]
    n_group = len(n_g)

    if chunksize == -1:
        chunksize = min(10, _auto_chunksize(n_group, processes))
    process_group = toolz.partial(
        _process_named_group_i, by=by, 
        apply_func=apply_func)
    labels_values = map(process_group, xrange(n_group), global_arg=n_g,
                  chunksize=chunksize, processes=processes,
                  use_pathos=use_pathos)
    result = _combine_apply_results(labels_values, by)    
    return result

def _combine_apply_results(labels_values, by):
    labels, values = zip(*labels_values)
    labels = list(labels)
    values = list(values)

    if isinstance(values[0], pd.DataFrame):
        result = pd.concat(values)
    else:
        result = pd.DataFrame(values)
    
    labels = np.repeat(
        labels, 
        [value.shape[0] if isinstance(value, pd.DataFrame) 
                        else 1
         for value in values],
        axis=0)

    if not isinstance(by, list):
        by = [by]
    for col in by:
        result[col] = None
    result.loc[:, col] = labels
    
    result.set_index(by, inplace=True)
    return result

def _process_named_group_i(i_group, n_g, by, apply_func):
    return _process_named_group(n_g[i_group], by, apply_func)

def _process_named_group((name, group), by, apply_func):
    try:
        single = apply_func(group)
        return name, single
    except KeyboardInterrupt as ex:
        print 'Keyboard interrupt'

def _auto_chunksize(nitems, processes):
    return (nitems / processes) + (nitems % processes>0)

def _groupby_apply_split(df, by, apply_func, processes=2,
                        chunksize=-1, use_pathos=False,
                        **kwargs):
    #timer = util.Timer()
    grouped = df.groupby(by, **kwargs)
    ngroups = grouped.ngroups
    default_chunk_size = 30
    if chunksize == -1:
        chunksize = min(default_chunk_size,
                        _auto_chunksize(ngroups, processes))
    sections = _split_groups(grouped, chunksize)
    #timer.stop('split groups')
    grouped = None

    worker = toolz.partial(_group_apply_dfsection, by=by, 
                           apply_func=apply_func,
                           **kwargs)
    results = map(worker, sections, global_arg=df,
                  processes=processes, chunksize=chunksize,
                  use_pathos=use_pathos)
    #timer.stop('process')

    if isinstance(results[0], pd.DataFrame):
        result = pd.concat(results)
    else:
        result = pd.DataFrame(results)
    #timer.stop('concat results')
    return result
        
def _group_apply_dfsection(section, df, by, 
                           apply_func, **gp_kwargs):
    #timer = util.Timer()
    try:
        grouped = df.iloc[section].groupby(by, **gp_kwargs) 
        #timer.stop('groupby')
        result =  grouped.apply(apply_func)
        #timer.stop('process section')
        return result
    except KeyboardInterrupt as ex:
        print 'Keyboard interrupt'

def _split_groups(grouped, chunksize):
    indices = grouped.indices
    group_indices = indices.values()
    ngroups = len(group_indices)
    return [list(toolz.concat(group_indices[igroup] 
                         for igroup in xrange(start, start+chunksize)
                         if igroup < ngroups))
            for start in xrange(0, ngroups, chunksize)]



if __name__ == '__main__':
        ''' Note: In this example, the parallel map is not faster than 
        non-parallel map or simply numpy.sum. This is for demonstrating example usage and testing correctness.
        
        A more realistic scenario I find good speedup with parallel map is when processing a big pandas.DataFrame
        (e.g., groupby then aggregation or apply), we let each worker process one part of the big dataframe, and 
        let the dataframe be the global_arg, which saves expensive copy operations, hence enjoying the speedup 
        from multiprocessing. See more discussions and another example at 
        [StackOverflow](http://stackoverflow.com/a/27683040/1100430)
        '''
        
        import numpy as np

        # Suppose we want to compute the sum of a large array
        big_array = np.random.rand(1e6, 100)
        
        # worker function that sums of a sub section of the array
        def section_sum(section, array):
            return array[section].sum()
        
        # split the big array into sections of 10000 rows, a worker sum up one section at a time.
        # To avoid expensive copy of the big array, pass it as the global_arg;
        # Pass indices of each array section as local_args to workers, which is not much data.
        section_size = 10000
        sections = [xrange(start, start+section_size) 
                    for start in xrange(0, big_array.shape[0], section_size)]
        # return a list of sum, one for each section
        section_sum_list = map(section_sum, sections, global_arg=big_array,
                               chunksize=25, processes=4)
        total_sum = sum(section_sum_list) # reduce results
        
        assert np.allclose(total_sum, big_array.sum())
