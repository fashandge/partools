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
import toolz
import string

def map(func, local_args, global_arg=None,
        chunksize=1, processes=2):
    ''' A parallel version of standard map function.
    
    This function is designed for infinitely parallelizable tasks on a single machine with multiple cores. 
    It has the following features:
    
    1. Ease of use. It can serve almost a drop-in replacement for standard non-parallel map function, while 
    magically exploiting the multiple cores in your box. That is, the worker function can be almost 
    arbitrary function, thanks to the pathos package (https://github.com/uqfoundation/pathos/blob/master/pathos)
    that uses dill package. You can use parmap.map anywhere in your source code, rather than just in the main 
    function.
    
    2. Using this function, you can avoid unnecessary copy of read-only large data structure. Suppose we want 
    to process a big pandas. DataFrame by sub sections using multiple cores. By default, the data structure will be     pickled if you pass it as an argument of the worker function. If the data structure is large, the additional 
    memory cost can be unaffordable, and the time for pickling large data structure can often make multiprocessing 
    slower than the single-threaded version. However, in certain cases, the children processes just read 
    different parts of the big data structure, do some processing and return some results. It is unncessary to copy
    the big data structure, which is also enabled by the copy-on-write mechanism of linux. The solution is to let 
    the big data structrue be a global variable of the calling module for multiprocessing, and do NOT pass the data
    structure directly as an argument for worker function. This function makes this solution tidy and transparent.
    
    args:
        func: a worker function that accepts either 1 (when global_arg is None) 
        or 2 (when global_arg is not None) arguments
        
        local_args: An iterable of data you have to copy to children processes, e.g., the array indices. The               cildren processes N=chunksize of them at a time.
        
        global_arg: the large object you want to share with children processes, but don't want to copy 
        to children process, like a big pandas dataframe, a large numpy array. The worker function should use 
        it as if it is read-only, otherwise expensive copy operation follows. 
        
        chunksize: number of items to be assigned to a child process at a time. Same as in the standard Pool.map.
        
        processes: number of children processes (workers) in the pool, set it up to the number of physical cores.
        Same as in the standard Pool.map. If processes=1, it is equivalent to non-parallel map.
        
     Example usages:
        See main function.
    '''
    
    global_arg_name = None
    try:
        if global_arg is not None:
            global_arg_name = random_string(10, prefix='_tmp_global_arg')
            # use a temporary global variable to hold the large object global_arg
            globals()[global_arg_name] = global_arg
            
            # wrap the original worker function using a temporary global variable name
            # so that we just pass a name (string) instead of the actual big object (global_arg) to 
            # the wrapped worker function
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

_charset = string.ascii_letters + string.digits
def random_string(length, prefix='', suffix=''):
    return '{}{}{}'.format(
        prefix+'_' if prefix else '',
        ''.join(random.sample(_charset, length)),
        '_'+suffix if suffix else ''
    )

if __name__ == '__main__':
        ''' Note: In this example, the parallel map is not faster than 
        non-parallel map or simply numpy.sum. This is for demonstrating example usage and testing correctness.
        
        A more realistic scenario I find good speedup with parallel map is when processing a big pandas.DataFrame
        (e.g., groupby then aggregation or apply), we let each worker process one part of the big dataframe, and 
        let the dataframe be the global_arg, which saves expensive copy operations, hence enjoying the speedup 
        from multiprocessing.
        '''
        
        import numpy as np

        # Suppose we want to compute the sum of a large array
        big_array = np.random.rand(1e6, 100)
        
        # worker function that sums of a sub section of the array
        def section_sum(rows, array):
            return array[rows].sum()
        
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
        
        assert(total_sum == big_array.sum())
