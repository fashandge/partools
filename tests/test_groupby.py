from unittest import TestCase
import numpy as np
import pandas as pd
import parmap
#from IPython.core.debugger import Tracer

class TestGroupby(TestCase):

    def test_equal(self):
        import numpy as np
        import pandas as pd
        from IPython.core.debugger import Tracer

        n_group = 100
        uniq_ids = range(1, n_group+1)
        size = 1e4

        ids = np.random.choice(uniq_ids, size)
        sigma = 0.2
        mu = 1.6
        scores = sigma * np.random.randn(size) + mu

        df = pd.DataFrame(dict(ids=ids, scores=scores))

        by_cols = ['ids']
        grouped = df.groupby(by_cols, sort=False)

        def mean_score(g):
            return pd.DataFrame(
                dict(scores=[g.scores.mean()])
            )

        result1 = parmap.dfg_apply(grouped, by_cols,
                                   mean_score, processes=4)
        result1.reset_index(inplace=True, drop=True)

        result2 = grouped.mean()
        result2.reset_index(inplace=True)

        self.assertTrue(all(result1 == result2))
