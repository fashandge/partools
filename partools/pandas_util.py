import pandas as pd
import numpy as np
import toolz
from . import parmap
from .config import *

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
    labels_values = parmap.map(process_group, grouped, 
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
    labels_values = parmap.map(process_group, xrange(n_group), global_arg=n_g,
                  chunksize=chunksize, processes=processes,
                  use_pathos=use_pathos)
    result = _combine_apply_results(labels_values, by)    
    return result

def _combine_apply_results(labels_values, by):
    labels = [label for label, _ in labels_values]
    values = [value for _, value in labels_values]

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
    results = parmap.map(worker, sections, global_arg=df,
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


