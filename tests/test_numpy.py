from unittest import TestCase
import numpy
import parmap

class TestParmap(TestCase):

    def test_equal(self):
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
        section_sum_list = parmap.map(section_sum, sections, global_arg=big_array,
                                      chunksize=25, processes=4)
        total_sum = sum(section_sum_list) # reduce results

        # test one process, equivalent to single-threaded map 
        section_sum_list2 = parmap.map(section_sum, sections, global_arg=big_array,
                                      processes=1)
        self.assertEqual(section_sum_list, section_sum_list2)
        total_sum2 = sum(section_sum_list2) # reduce results
        self.assertEqual(total_sum, total_sum2)
        
        self.assertTrue(np.allclose(total_sum, big_array.sum()))
