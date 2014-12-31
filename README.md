partools
=============

Utilitiy functions for python parallel processing for big data structure with multicores on a single machine. Avoid unnecessary copy of read-only large data structure. It includes: 1. A parallel version of `map` function. It is designed for infinitely parallelizable tasks on a single machine with multiple cores, in a memory-efficient way. It can drop-in replace the standard map function in most cases, and harness the power of multiple cores. 2. utility function for parallel processing of the popular pandas dataframes. The package is based on the pathos and toolz packages. It has the following features:
 
 +  **Ease of use**. It can serve almost a drop-in replacement for the standard non-parallel map function, while magically exploiting the multiple cores in your box. Optionally, if you encounter worker function pickling issue, set use\_pathos=True (require [pathos package](https://github.com/uqfoundation/pathos/blob/master/pathos) that uses dill package). 
 
 +   **Avoid unnecessary copy of a read-only large data structure**. Suppose we want to process a big object, e.g., a pandas.DataFrame, by parts using multiple cores. By default, the data structure will be pickled if you pass it as an argument of the worker function. If the data structure is large, the additional memory cost can be unaffordable, and the time for pickling large data structure can often make multiprocessing slower than the single-threaded version. 
 
 However, in many scenarios, the children processes just read different parts of the big data structure, do some processing and return some results. It is unnecessary to copy the big data structure, which is also enabled by the copy-on-write mechanism of linux (but not in windows). The solution is to let the big data structrue be a temporary *global* variable of the calling module for multiprocessing, and do NOT pass the data  structure directly as an argument for worker function. This function encapsulates all those messy details so that we use it as if it is the standard map function with a few additional options to exploit multiple cores.
    
###Example usages:
Currently the package has two functions: (1) `map`, for general data processing. (2) `groupby_apply`, for pandas dataframe grouping and parallel processing of groups.

\* Note: In this example, the parallel map is not faster than non-parallel map or simply numpy.sum. This is for demonstrating example usage and testing correctness. 

See more discussions and a more realistic scenario on parallel processing of pandas data frame at [StackOverflow](http://stackoverflow.com/a/27683040/1100430)
        
```python
import numpy as np
import partools
        
# Suppose we want to compute the sum of a large array
big_array = np.random.rand(1e6, 100)

# worker function that sums of a sub section of the array
def section_sum(section, array):
    return array[section].sum()
        
# split the big array by rows, each worker sum up one section of 10000 rows at a time
section_size = 10000
sections = [xrange(start, start+section_size) 
            for start in xrange(0, big_array.shape[0], section_size)]

# To avoid expensive copy of the big array, set it as the global_arg. The key assumption
# is that the worker function will NOT modify the big array (read-only).
section_sum_list = partools.map(section_sum, sections, global_arg=big_array,
                              chunksize=25, processes=4)
total_sum = sum(section_sum_list) # reduce results

assert(total_sum == big_array.sum())
```

### Dependencies
Tested with python 2.7 in linux with the following packages:
+ [toolz 0.6.0](https://pypi.python.org/pypi/toolz)
+ cPickle
+ [pathos 0.2a1.dev](http://danse.cacr.caltech.edu/packages/dev_danse_us/pathos-0.2a.dev-20130811.zip) \[optional\] ([installation guide](http://trac.mystic.cacr.caltech.edu/project/pathos/wiki/Installation))
+ numpy and pandas, if you want to process pandas dataframe in parallel
