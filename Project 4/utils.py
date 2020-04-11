from itertools import product


# Generates config dicts for each possible combination of incoming arguments
def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))
