import numpy as np
import pandas as pd
develop = False
if develop:
    import src.parmap.parmap as parmap
    from IPython.core.debugger import Tracer
else:
    import parmap

# settings
processes = 4
chunksize = -1
use_pathos = False

# applyfunc returns a dataframe
def mean_score(g):
    return pd.DataFrame(
        dict(scores=[g.scores.mean()])
    )

# applyfunc returns a series 
def mean_score_series(g):
    return pd.Series(
        dict(scores=g.scores.mean())
    )

def prepare_df():
    n_group = 100
    uniq_ids = range(1, n_group+1)
    size = 1e4

    ids = np.random.choice(uniq_ids, size)
    sigma = 0.2
    mu = 1.6
    scores = sigma * np.random.randn(size) + mu

    df = pd.DataFrame(dict(ids=ids, scores=scores))
    by = 'ids'
    result0 = df.groupby(by).apply(mean_score_series)
    return df, by, result0

def test_iter():
    df, by, result0 = prepare_df()
    result = parmap.groupby_apply(
        df, by, mean_score_series, 
        algorithm='default',
        processes=processes,
        chunksize=chunksize,
        use_pathos=use_pathos
    )
    assert all(result0==result)

def test_iter_local():
    df, by, result0 = prepare_df()
    result = parmap.groupby_apply(
        df, by, mean_score_series, 
        algorithm='iter_local',
        processes=processes,
        chunksize=chunksize,
        use_pathos=use_pathos
    )
    assert all(result0==result)

def test_split():
    df, by, result0 = prepare_df()
    result = parmap.groupby_apply(
        df, by, mean_score_series, 
        algorithm='split',
        processes=processes,
        chunksize=chunksize,
        use_pathos=use_pathos
    )
    assert all(result0==result)

