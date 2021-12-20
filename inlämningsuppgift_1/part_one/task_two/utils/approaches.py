import random


def random_path(domain: list[int], func) -> int:
    """Calls fitness function with a shuffled list of domain"""
    return func(random.sample(domain, len(domain)))
