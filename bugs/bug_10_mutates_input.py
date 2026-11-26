"""Return a new list with a bonus added to each score.

The original list must not be changed.
"""


def with_bonus(scores, bonus):
    out = scores
    for i in range(len(out)):
        out[i] += bonus
    return out
