import numpy as np
import scipy.sparse as ss

def try_desparsify(csr_arr):
    """
    util to get reg arr back from sparse array

    with a check to avoid typeErr in case input is np
    """
    try:
        out = csr_arr.toarray()
        return out 
    except:
        return csr_arr