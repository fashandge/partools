import numpy as np
import pandas as pd
from ..pandas_util import groupby_apply, series_apply
from ..parmap import partial
import toolz

# settings
processes = 4
chunksize = -1
use_pathos = 'auto'
_intercept = 3

# applyfunc returns a dataframe
def mean_score(g):
    return pd.DataFrame(
        dict(scores=[g.scores.mean()])
    )

# applyfunc returns a series 
def mean_score_series_with_global(g, intercept):
    return pd.Series(
        dict(scores=g.scores.mean()+intercept)
    )

def mean_score_series(g):
    return mean_score_series_with_global(g, _intercept)
    

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
    # apply doesn't accept toolz.partial, but accepts parmap.partial
    result0 = df.groupby(by).apply(
        partial(mean_score_series_with_global, intercept=_intercept))
    return df, by, result0

def times_hundred_with_global(score, intercept):
    return score * 10 + intercept

def times_hundred(score):
    return score * 10

def test_series_apply():
    df, by, result0 = prepare_df()
    _result = df.scores.apply(times_hundred)
    result = series_apply(
        df.scores, times_hundred,
        processes=processes,
        use_pathos=use_pathos
    )
    assert all(_result==result)

def test_series_apply_with_global():
    df, by, result0 = prepare_df()
    _result = df.scores.apply(
        partial(times_hundred_with_global, intercept=_intercept))
    result = series_apply(
        df.scores, times_hundred_with_global,
        global_arg=_intercept,
        processes=processes,
        use_pathos=use_pathos
    )
    assert all(_result==result)

def test_vanilla():
    df, by, result0 = prepare_df()
    result = groupby_apply(
        df, by, mean_score_series_with_global, 
        global_arg = _intercept,
        processes=1)
    assert all(result0==result)

def test_iter():
    df, by, result0 = prepare_df()
    result = groupby_apply(
        df, by, mean_score_series, 
        algorithm='default',
        processes=processes,
        chunksize=chunksize,
        use_pathos=use_pathos
    )
    assert all(result0==result)

def test_iter_local():
    df, by, result0 = prepare_df()
    result = groupby_apply(
        df, by, mean_score_series, 
        algorithm='iter_local',
        processes=processes,
        chunksize=chunksize,
        use_pathos=use_pathos
    )
    assert all(result0==result)

def test_split():
    df, by, result0 = prepare_df()
    result = groupby_apply(
        df, by, mean_score_series, 
        algorithm='split',
        processes=processes,
        chunksize=chunksize,
        use_pathos=use_pathos
    )
    assert all(result0==result)

def test_sort():
    df, by, result0 = prepare_df()
    result = groupby_apply(
        df, by, mean_score_series, 
        algorithm='sort',
        processes=processes,
        chunksize=chunksize,
        use_pathos=use_pathos
    )
    assert all(result0==result)

def test_sort_with_global():
    df, by, result0 = prepare_df()
    result = groupby_apply(
        df, by, mean_score_series_with_global, 
        global_arg=_intercept,
        algorithm='sort',
        processes=processes,
        chunksize=chunksize,
        use_pathos=use_pathos
    )
    assert all(result0==result)
