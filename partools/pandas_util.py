import pandas as pd
import numpy as np
import toolz
from . import parmap
from . import global_manager as gm
from .config import *

def groupby_apply(df, by, func, 
                  global_arg=None,
                  use_agg=False,
                  algorithm='default', 
                  processes=2, chunksize=-1,
                  use_pathos='auto', 
                  keep_order=True, no_sort=False,
                  **groupby_kwargs):
    '''Group a pandas dataframe and process groups in parallel, similar to df.groupby().apply

    Parameters
    ----------
    df : pandas.DataFrame
    by : string or a list of string
        column names to groupby
    func : function or a hash table
        apply function, or object can be passed to grouped.agg, e.g., a hash table
        (column name => func) (the latter is valid only when algorithm=='sort' or 'split'.
        if global_arg=None, the apply func func(local_arg) should accepts a single argument;
        if global_arg is not None, the apply func func(local_arg, global_arg) should accepts two arguments.
    global_arg : object
        the large object you want to share with children processes, but don't want to copy 
        to children process, like a big pandas dataframe, or a large numpy array. The worker function should use 
        it as if it is read-only, otherwise expensive copy operation follows. 
    use_agg : boolean, default False
        Only applicable to 'split' and 'sort'. if True, call grouped.agg(func), 
        otherwise func(group). It is faster than the more general apply.
    algorithm : string, default 'default' (i.e., "sort")
        one of 'iter', 'iter_local', 'split', or 'sort'; 
        specify which algorithm to use. Leave it as default most of the time.
        Or set it to 'split', can be faster than 'iter' and 'iter_local';
        'sort' can be fastest if sorting df is quick. 'iter_local' is usually 
        the slowest one.
    processes : int, default 2
        Number of processes. If processes=1, directly call
        df.groupby(by, **groupby_kwargs).apply(func) [or .agg(func), if use_agg=True], 
        other parameters have no effect.
    chunksize : int, default -1
        use -1 to choose chunksize automatically 
    keep_order, no_sort : boolean
        applicable only when algorithm=='sort'
    groupby_kwargs : optional keyword arguments
        optional keyword arguments for df.groupby

    Returns
    -------
    A dataframe : usually the same as non-parallel version
    '''
    try:
        # the pandas.DataFrame.groupby().apply doesn't accept toolz.partial function
        # so use our own implementation parmap.partial
        process_func, global_arg_name = parmap.wrap_global_arg(
            func, global_arg, use_toolz=False)

        if processes == 1: # no-parallel processing
            return _vanilla_groupby_apply(
                df, by, process_func, use_agg=use_agg, 
                **groupby_kwargs)

        #first sort then split the df into sections, and process each section in parallel
        #This can be the fastest version of groupby_apply*, if sorting df is quick.
        if algorithm == 'default' or algorithm == 'sort':
            alg = _groupby_apply_sort

        # first groupby df, then split df into sections each of which has
        # multiple groups. Applying over multiple groups together is potentiall
        # faster than applying over invidual groups one by one.
        # 
        elif algorithm == 'split':
            alg = _groupby_apply_split

        # first groupby df, iterate and process each group
        # pass position of groups to worker function, hence avoid copying
        # the DataFrameGroupby object and groups
        elif algorithm=='iter':
            alg = _groupby_apply_iter

        # first groupby df, iterate and process each group
        # directly iterate DataFrameGroupby and pass groups to worker function
        elif algorithm == 'iter_local':
            alg = _groupby_apply_iter_local

        else:
            raise Exception('algorithm {} is not supported. Use one'
                            'of ["sort"(default), "split", "iter", "iter_local"]')

        if global_arg is not None and use_pathos=='auto':
            use_pathos = True

        result = alg(df, by, process_func,
                     use_agg=use_agg,
                     processes=processes, chunksize=chunksize,
                     use_pathos=use_pathos,
                     **groupby_kwargs)
        return result
    finally:
        gm.remove_global_arg(global_arg_name)

def series_apply(series, func, global_arg=None,
                 processes=2, chunksize=-1, 
                 use_pathos='auto', **apply_kwargs):
    '''parallel series apply, similar to series.apply(func)'''
    try:
        process_func, global_arg_name = parmap.wrap_global_arg(
            func, global_arg, use_toolz=False)

        if processes == 1:
            return series.apply(process_func, **apply_kwargs)

        #if global_arg is not None and use_pathos=='auto':
        #    use_pathos = True

        if chunksize == -1:
            chunksize = 1
        sections = _auto_chunks(series.shape[0], processes)
        worker = toolz.partial(_apply_series_section, 
                               func=process_func, **apply_kwargs)
        result = parmap.map(
            worker,
            sections, 
            global_arg=series,
            processes=processes,
            chunksize=chunksize,
            use_pathos=use_pathos
        )
        result = pd.concat(result)
        return result
    finally:
        gm.remove_global_arg(global_arg_name)

def _apply_series_section(section, series, func, **apply_kwargs):
    return series[section].apply(func, **apply_kwargs)

def _auto_chunks(size, nchunks):
    chunk_size = size/nchunks + ((size%nchunks) > 0)
    return [slice(start, start+chunk_size)
            for start in xrange(0, size, chunk_size)]

def _auto_chunksize(nitems, processes):
    return (nitems / processes) + (nitems % processes>0)

def _groupby_apply_sort(df, by, func, 
                        use_agg=False, 
                        processes=2, chunksize=-1, 
                        use_pathos=False,
                        keep_order=True, no_sort=False,
                        **kwargs):
    '''first sort then split the df into sections, and process each section in parallel

    This can be the fastest version of groupby_apply*, if sorting df is quick.

    parameters
    ----------
    keep_order : boolean, default True
        whether to keep the original order of df after processing
    no_sort : boolean, default False
        if df is already sorted, no need to sort again
    '''
    # sort df by groupby columns
    order_col = None
    if not no_sort:
        order_col = gm._random_string(10, prefix='_order')
        df[order_col] = np.arange(df.shape[0])
        df.sort(by, inplace=True)

    if chunksize == -1:
        chunksize = 1
    sections = _split_df_by_groups(df, by, processes)

    worker = toolz.partial(_group_apply_dfsection, by=by, 
                           func=func, use_agg=use_agg,
                           **kwargs)
    result = parmap.map(
        worker,
        sections, 
        global_arg=df,
        processes=processes,
        chunksize=chunksize,
        use_pathos=use_pathos
    )
    result = pd.concat(result)

    # recover order
    if not no_sort:
        if keep_order:
            df.sort(order_col, inplace=True)
        df.drop(order_col, axis=1, inplace=True)
    return result

def _groupby_apply_split(df, by, func, use_agg=False,
                         processes=2, chunksize=-1, 
                         use_pathos=False,
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
                           func=func, use_agg=use_agg,
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
                           func, use_agg=False, 
                           **gp_kwargs):
    grouped = df.iloc[section].groupby(by, **gp_kwargs) 
    if use_agg:
        result = grouped.agg(func)
    else:
        result =  grouped.apply(func)
    return result

def _split_groups(grouped, chunksize):
    indices = grouped.indices
    group_indices = indices.values()
    ngroups = len(group_indices)
    return [
        list(toolz.concat(
            group_indices[slice(start, min(start+chunksize, ngroups))]
            ))
            for start in xrange(0, ngroups, chunksize)]

def _split_df_by_groups(df, by, n_section):
    size = df.shape[0]
    section_size = size / n_section
    sections = []
    start = 0
    by_values = df[by]
    for i in range(n_section-1):
        stop = start + section_size
        while stop<size and by_values.iloc[stop-1]==by_values.iloc[stop]:
            stop += 1
        sections.append(slice(start, stop))
        start = stop
    sections.append(slice(start, size))
    return sections

def _vanilla_groupby_apply(df, by, func, use_agg=False, 
                           **groupby_kwargs):
    grouped = df.groupby(by, **groupby_kwargs)
    if use_agg:
        return grouped.agg(func)
    else:
        return grouped.apply(func)

def _groupby_apply_iter_local(df, by, func, use_agg=False, 
                              processes=2, chunksize=-1, 
                              use_pathos=False,
                              **groupby_kwargs):
    grouped = df.groupby(by, **groupby_kwargs)
    if chunksize == -1:
        default_chunksize = 10
        chunksize = min(default_chunksize, 
                        _auto_chunksize(grouped.ngroups, processes))
    process_group = toolz.partial(
        _process_named_group, by=by, 
        func=func)
    labels_values = parmap.map(process_group, grouped, 
                               processes=processes,
                               chunksize=chunksize,
                               use_pathos=use_pathos)
    result = _combine_apply_results(labels_values, by)    
    return result

def _groupby_apply_iter(df, by, func, 
                        use_agg=False,
                        processes=2,
                        chunksize=-1, use_pathos=False,
                        **groupby_kwargs):
    grouped = df.groupby(by, **groupby_kwargs)
    n_g = [(name, group) for name, group in grouped]
    n_group = len(n_g)

    if chunksize == -1:
        chunksize = min(10, _auto_chunksize(n_group, processes))
    process_group = toolz.partial(
        _process_named_group_i, by=by, 
        func=func)
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

def _process_named_group_i(i_group, n_g, by, func):
    return _process_named_group(n_g[i_group], by, func)

def _process_named_group((name, group), by, func):
    try:
        single = func(group)
        return name, single
    except KeyboardInterrupt as ex:
        print 'Keyboard interrupt'
