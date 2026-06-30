import random
from typing import Tuple


def sample_lambda(clean_prob: float, dirty_min: float, dirty_max: float) -> Tuple[float, bool]:
    if random.random() < clean_prob:
        return 1.0, True
    return random.uniform(dirty_min, dirty_max), False
