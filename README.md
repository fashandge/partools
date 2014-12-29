python_parmap
=============

A parallel version of map function is designed for infinitely parallelizable tasks on a single machine with multiple cores. It depends on the pathos and toolz packages.
    It has the following features:
 
 +  **Ease of use**. It can serve almost a drop-in replacement for the standard non-parallel map function, while magically exploiting the multiple cores in your box. That is, the worker function can be almost arbitrary function, thanks to the pathos package (https://github.com/uqfoundation/pathos/blob/master/pathos) that uses dill package. You can use parmap.map anywhere in your source code, rather than just in the main  function.
 
 +   **Avoid unnecessary copy of read-only large data structure**. Suppose we want to process a big object, e.g., a pandas.DataFrame, by parts using multiple cores. By default, the data structure will be     pickled if you pass it as an argument of the worker function. If the data structure is large, the additional memory cost can be unaffordable, and the time for pickling large data structure can often make multiprocessing slower than the single-threaded version. 
 
 However, in certain cases, the children processes just read different parts of the big data structure, do some processing and return some results. It is unnecessary to copy the big data structure, which is also enabled by the copy-on-write mechanism of linux. The solution is to let the big data structrue be a *global* variable of the calling module for multiprocessing, and do NOT pass the data  structure directly as an argument for worker function. This function makes this solution tidy and transparent.
    
###Example usages:
\* Note: In this example, the parallel map is not faster than non-parallel map or simply numpy.sum. This is for demonstrating example usage and testing correctness. 

See more discussions and a more realistic scenario on parallel processing of pandas data frame at [StackOverflow](http://stackoverflow.com/a/27683040/1100430)
        
```python
import numpy as np
import parmap
        
# Suppose we want to compute the sum of a large array
big_array = np.random.rand(1e6, 100)

# worker function that sums of a sub section of the array
def section_sum(section, array):
    return array[section].sum()
        
# split the big array by rows, each worker sum up one section of 10000 rows at a time
# To avoid expensive copy of the big array, set it as the global_arg
section_size = 10000
sections = [xrange(start, start+section_size) 
            for start in xrange(0, big_array.shape[0], section_size)]
            
section_sum_list = parmap.map(section_sum, sections, global_arg=big_array,
                              chunksize=25, processes=4)
total_sum = sum(section_sum_list) # reduce results

assert(total_sum == big_array.sum())
```
