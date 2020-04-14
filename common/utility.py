# Author: Mattia Silvestri


def calc_qvals(rewards, gamma):
    """
    Compute discouted rewards-to-go.
    :param rewards: episode rewards; as list
    :param gamma: discount factor; as double
    :return: expected Q-values; as list
    """
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r += r * gamma
        res.append(sum_r)

    return list(reversed(res))