import numpy as np


def cluster_onsets(onsets):
    '''Returns sorted clusters of onsets.

    Does not asume onsets are cyclical.

    Args:
        onsets: list of ms

    Returns:
        List of ints where each int represents how many onsets should be in that
        cluster.
        The sum of the return value should be equal to the length of the
        'onsets' list.
    '''

    max_clustering_dur = 450  # ms

    iois = np.diff(onsets)
    iois_set = list(set(iois))
    median_ioi = sorted(iois_set)[len(iois_set) / 2]
    clusters = [1]
    for idx in xrange(1, len(onsets) - 1):
        prev_dur = iois[idx - 1]
        next_dur = iois[idx]
        min_dur = min(prev_dur, next_dur)
        if (min_dur == prev_dur and min_dur < median_ioi and
                min_dur < max_clustering_dur):
            clusters[-1] += 1
        else:
            clusters.append(1)

    if next_dur < median_ioi and max_clustering_dur:
        clusters[-1] += 1
    else:
        clusters.append(1)

    return clusters


def accented_onsets(onsets):
    'Returns subset of onsets that are rhythmically accented (Povel 1985)'
    clusters = cluster_onsets(onsets)
    accented = []
    it = iter(onsets)
    for cluster in clusters:
        if cluster == 1:
            accented.append(it.next())
        elif cluster == 2:
            it.next()
            accented.append(it.next())
        else:
            accented.append(it.next())
            cluster -= 2
            while cluster != 0:
                it.next()
                cluster -= 1
            accented.append(it.next())
    return accented


def hypothesis_counterevidence(onsets, hypothesis, W=4):
    '''
    Calculates the counter evidence score in Povel 1985.

    Args:
        onsets: list of time onsets
        hypothesis: (phase, period) tuple. Period and onset times should be
            multples of a same base time step
        W: weight of -ev counterevidence (see paper)

    Returns:
        int with counterevidence score
    '''
    accents = accented_onsets(onsets)
    counterevidence = 0
    projection = hypothesis[0]
    while projection < onsets[-1] + hypothesis[1]:
        if projection not in accents:
            if projection not in onsets:
                counterevidence += W
            else:
                counterevidence += 1
        projection += hypothesis[1]
    return counterevidence


def best_clock(onsets, base_time_step, phrase_length):
    '''
    Returns the best clock (phase, period) for the onset sequence using an
    implementation of Povel 1985's model.

    Args:
        onsets: list of onset times (in ms or multiple of base time step)
        base_time_step: length in milliseconds of base timestep or 1 if
            onsets are not milliseconds
        phrase_length: length in base steps of a phrase

    Returns:
        ((phase, period), counterevidence)
    '''
    hypothesis_space = [
        (phase, period)
        for period in np.arange(1, phrase_length / 2) * base_time_step
        for phase in np.arange(0, period)
        if phrase_length % period == 0
    ]
    hypothesis_space_w_score = [(hypothesis,
                                 hypothesis_counterevidence(onsets, hypothesis))
                                for hypothesis in hypothesis_space]
    sorted_hs = sorted(hypothesis_space_w_score, key=lambda x: x[1])
    return sorted_hs[0]


def cv_to_category(counterevidence):
    return counterevidence + 1
