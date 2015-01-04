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
import toolz
import multiprocessing as mp_std
try:
    import pathos.multiprocessing as mp_pathos
except:
    mp_pathos = None
from . import global_manager as gm
from .config import *

# Used as the timeout
GOOGLE = 1e100

def map(func, iterable, global_arg=None,
        chunksize=1, processes=2, use_pathos='auto'):
    ''' A parallel version of standard map function. It is a blocking operation and return ordered results.

    If you want to process a read-only large data structure by parts in parallel, put it in the global_arg. 
    This function helps to avoid unncessary copy of that large data to be memory and computaton efficient.
    This is enabled by the copy-and-write feature of fork in Linux (but not in windows).

    Parameters
    ----------
    func : worker function func with one or two arguments
        1. if global_arg is None: func(local_arg) should accept one argument
        2. if global_arg is not None: func(local_arg, global_arg) should accept two arguments
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

    use_pathos: boolean or string 'auto', default 'auto'
        use pathos.multiprocessing or standard multiprocessing package. Recommend to use latter most of the time
        as it is faster in general. Only when your function cannot be pickled will you consider use pathos. If set
        to 'auto', the function will set use_pathos=False if it can pickle the function, otherwise True.

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
    if processes != 1:
        use_pathos = _determine_pathos_usage(func, use_pathos)

    try:
        pool = None
        process_func, global_arg_name = \
            wrap_global_arg(
                func, global_arg
            )

        if processes == 1:
            import __builtin__
            result = __builtin__.map(process_func, iterable)
        else:
            if use_pathos and mp_pathos is None:
                raise Exception('pathos package not available, cannot use_pathos=Ture')
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
        gm.remove_global_arg(global_arg_name)
        _terminate_pool(pool)

def wrap_global_arg(func, global_arg, use_toolz=True):
    if global_arg is None:
        process_func = func
        global_arg_name = None
    else:
        process_func, global_arg_name = \
            _create_func_with_global(
                func, global_arg, 
                use_toolz=use_toolz
            )
    return process_func, global_arg_name

def _determine_pathos_usage(func, use_pathos):
    if use_pathos == False:
        _trypickle(func)
    elif use_pathos == 'auto':
        try:
            _trypickle(func, silent=True)
        except:
            use_pathos = True
        else:
            use_pathos = False
    return use_pathos

def _terminate_pool(pool):
    if pool is not None:
        pool.close()
        pool.terminate()
        pool.join()

def _create_func_with_global(func, global_arg, use_toolz=True):
    # use a temporary global variable to hold the large object global_arg
    global_arg_name = gm.set_global_arg(global_arg)

    # partial function can be pickled by multiprocessing
    # but not inner function.
    _partial = toolz.partial if use_toolz else partial
    process_func = _partial(
        _func_with_global,
        global_arg_name=global_arg_name,
        func=func
    )
    return process_func, global_arg_name

# if some library doesn't accept toolz.partial, (say pd.dataframe.groupby().apply)
# try this implementation for paritial
def partial(_function, *args, **keywords):
    def newfunc(*fargs, **fkeywords):
        newkeywords = keywords.copy()
        newkeywords.update(fkeywords)
        return _function(*(args + fargs), **newkeywords)
    newfunc.func = _function
    newfunc.args = args
    newfunc.keywords = keywords
    return newfunc

# wrap the original worker function using a temporary global variable name
# so that we just pass a name (string) instead of the actual big object (global_arg) to 
# the wrapped worker function
def _func_with_global(local_arg, global_arg_name, func):
    try:
        #global_arg = globals()[global_arg_name] 
        global_arg = gm.get_global_arg(global_arg_name)
        return func(local_arg, global_arg)
    except KeyboardInterrupt:
        print 'KeyboardInterrupt'

def _trypickle(func, silent=False):
    """
    Attempts to pickle func since multiprocessing needs to do this.

    copied from Rosseta package.
    """
    genericmsg = ("Pickling of func (necessary for multiprocessing) failed. "
                  "Please refactor the function to be a top-level function in "
                  "a module, or set use_pathos=True if you have pathos installed.")

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
        if not silent:
            if 'instancemethod' in e.message:
                sys.stderr.write(boundmethodmsg + "\n")
            else:
                sys.stderr.write(genericmsg + '\n')
        raise
    except:
        if not silent:
            sys.stderr.write(genericmsg + '\n')
        raise

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
