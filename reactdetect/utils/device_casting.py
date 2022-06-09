import numpy as np 
import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

"""
expensive operation to also check device of iterable other than list

use causiously, there is a reason allennlp has a impl
    that does not support many iterables
"""

def has_tensor(obj) -> bool:
    """
    Given a possibly complex data structure,
    check if it has any torch.Tensors in it.
    """
    if isinstance(obj, torch.Tensor):
        return True
    elif isinstance(obj, dict):
        return any(has_tensor(value) for value in obj.values())
    elif isinstance(obj, (list, tuple)):
        return any(has_tensor(item) for item in obj)
    else:
        return False


import numpy as np 
import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

"""
expensive operation to also check device of iterable other than list

use causiously, there is a reason allennlp has a impl
    that does not support many iterables
"""

def has_tensor(obj) -> bool:
    """
    Given a possibly complex data structure,
    check if it has any torch.Tensors in it.
    """
    if isinstance(obj, torch.Tensor):
        return True
    elif isinstance(obj, dict):
        return any(has_tensor(value) for value in obj.values())
    elif isinstance(obj, (list, tuple)):
        return any(has_tensor(item) for item in obj)
    else:
        return False


def move_to_cpu(obj):
    """
    Poor ctrl+v wrapper around allennlp move_to_device function

    relatively expensive operation but works
    """
    from allennlp.common.util import int_to_device

    cuda_device = torch.device('cpu')

    if isinstance(obj, torch.Tensor):
        return obj.cuda(cuda_device)
    elif isinstance(obj, dict):
        return {key: move_to_cpu(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cpu(item) for item in obj]
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # This is the best way to detect a NamedTuple, it turns out.
        return obj.__class__(*(move_to_cpu(item) for item in obj))
    elif isinstance(obj, tuple):
        return tuple(move_to_cpu(item) for item in obj)
    elif type(obj) ==  np.ndarray:
        if obj.dtype == np.ndarray:
            return  obj.astype(np.float64)
        else:
            return obj
    else:
        print('else')
        return obj