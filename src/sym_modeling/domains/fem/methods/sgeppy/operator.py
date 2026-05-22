import numpy as np
import operator # from geppy

EPS = 1e-12

def add(a, b):
    """Element-wise addition."""
    return operator.add(a, b)

def sub(a, b):
    """Element-wise subtraction."""
    return operator.sub(a, b)

def mul(a, b):
    """Element-wise multiplication."""
    return operator.mul(a, b)

def div(a, b):
    """Safe division that avoids division by zero."""
    if np.isscalar(b):
        if abs(b) < 1e-6:
            b = 1
    else:
        b[abs(b) < 1e-6] = 1
    return a / b

def neg(a):
    return -a

def square(a):
    return np.square(a)

def sqrt(a):
    return np.sqrt(np.abs(a) + EPS)

def log(a):
    return np.log(np.abs(a) + EPS)

def exp(a):
    return np.exp(np.clip(a, -20.0, 20.0))

def sin(a):
    return np.sin(a)

def cos(a):
    return np.cos(a)
