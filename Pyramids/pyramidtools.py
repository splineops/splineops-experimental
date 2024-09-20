# pyramidtools.py

import numpy as np
from pyramidfilters import (
    pyramid_filter_spline_l2,
    pyramid_filter_spline_L2,
    pyramid_filter_centered,
    pyramid_filter_centered_L2,
    pyramid_filter_centered_L2_derivative
)

def get_pyramid_filter(filter_name, order):
    """
    Get the coefficients of the filter (reduce and expand filter)
    Returns the coefficients in g and h, and IsCentered flag
    """
    is_centered = False

    if filter_name == "Spline":
        g, h = pyramid_filter_spline_l2(order)
        is_centered = False
    elif filter_name == "Spline L2":
        g, h = pyramid_filter_spline_L2(order)
        is_centered = False
    elif filter_name == "Centered Spline":
        g, h = pyramid_filter_centered(order)
        is_centered = True
    elif filter_name == "Centered Spline L2":
        g, h = pyramid_filter_centered_L2(order)
        is_centered = True
    else:
        raise ValueError("This family of filters is unknown")
    return g, h, is_centered

def reduce_1D(signal, g, is_centered):
    if is_centered:
        return reduce_centered_1D(signal, g)
    else:
        return reduce_standard_1D(signal, g)

def expand_1D(signal, h, is_centered):
    if is_centered:
        return expand_centered_1D(signal, h)
    else:
        return expand_standard_1D(signal, h)

def reduce_standard_1D(In, g):
    n = len(In)
    nred = n // 2
    ng = len(g)
    Out = np.zeros(nred)
    n2 = 2 * (n - 1)
    if ng < 2:
        # Simple average
        Out = (In[::2] + In[1::2]) / 2
    else:
        for kk in range(nred):
            k = 2 * kk
            Out[kk] = In[k] * g[0]
            for i in range(1, ng):
                i1 = k - i
                i2 = k + i
                if i1 < 0:
                    i1 = (-i1) % n2
                    if i1 > n - 1:
                        i1 = n2 - i1
                if i2 > n - 1:
                    i2 = i2 % n2
                    if i2 > n - 1:
                        i2 = n2 - i2
                Out[kk] += g[i] * (In[i1] + In[i2])
    return Out

def expand_standard_1D(In, h):
    n = len(In)
    nexp = n * 2
    nh = len(h)
    Out = np.zeros(nexp)
    n2 = n - 1
    if nh < 2:
        # Simple duplication
        Out[::2] = In
        Out[1::2] = In
    else:
        for i in range(nexp):
            for k in range(i % 2, nh, 2):
                i1 = (i - k) // 2
                if i1 < 0:
                    i1 = (-i1) % n2
                    if i1 > n2:
                        i1 = n2 - i1
                Out[i] += h[k] * In[i1]
            for k in range(2 - (i % 2), nh, 2):
                i2 = (i + k) // 2
                if i2 > n2:
                    i2 = i2 % n2
                    i2 = n2 - i2
                    if i2 > n2:
                        i2 = n2 - i2
                Out[i] += h[k] * In[i2]
    return Out

def reduce_centered_1D(In, g):
    n = len(In)
    nred = n // 2
    ng = len(g)
    n2 = 2 * n
    y_tmp = np.zeros(n)
    for k in range(n):
        y_tmp[k] = In[k] * g[0]
        for i in range(1, ng):
            i1 = k - i
            i2 = k + i
            if i1 < 0:
                i1 = (2 * n - 1 - i1) % n2
                if i1 >= n:
                    i1 = n2 - i1 - 1
            if i2 > n - 1:
                i2 = i2 % n2
                if i2 >= n:
                    i2 = n2 - i2 - 1
            y_tmp[k] += g[i] * (In[i1] + In[i2])
    Out = (y_tmp[::2] + y_tmp[1::2]) / 2
    return Out

def expand_centered_1D(In, h):
    n = len(In)
    nexp = n * 2
    nh = len(h)
    Out = np.zeros(nexp)
    n2 = 2 * n
    k0 = (nh // 2) * 2 - 1
    for i in range(n):
        j = i * 2
        Out[j] = In[i] * h[0]
        for k in range(2, nh, 2):
            i1 = i - k // 2
            i2 = i + k // 2
            if i1 < 0:
                i1 = (2 * n - 1 - i1) % n2
                if i1 >= n:
                    i1 = n2 - i1 - 1
            if i2 >= n:
                i2 = i2 % n2
                if i2 >= n:
                    i2 = n2 - i2 - 1
            Out[j] += h[k] * (In[i1] + In[i2])
        Out[j + 1] = 0.0
        for k in range(-k0, nh, 2):
            kk = abs(k)
            i1 = i + (k + 1) // 2
            if i1 < 0:
                i1 = (2 * n - 1 - i1) % n2
                if i1 > n - 1:
                    i1 = n2 - i1 - 1
            if i1 >= n:
                i1 = i1 % n2
                if i1 >= n:
                    i1 = n2 - i1 - 1
            Out[j + 1] += h[kk] * In[i1]
    # Apply inverse Haar filter
    for j in range(nexp - 1, 0, -1):
        Out[j] = (Out[j] + Out[j - 1]) / 2.0
    Out[0] /= 2.0
    return Out

def reduce_2D(In, g, is_centered):
    NxIn, NyIn = In.shape
    NxOut = NxIn // 2 if NxIn > 1 else 1
    NyOut = NyIn // 2 if NyIn > 1 else 1
    # Temporary array for processing
    Tmp = np.zeros((NyIn, NxOut))
    # X processing
    if NxIn > 1:
        for ky in range(NyIn):
            InBuffer = In[ky, :]
            OutBuffer = reduce_1D(InBuffer, g, is_centered)
            Tmp[ky, :] = OutBuffer
    else:
        Tmp = In.copy()
    # Y processing
    Out = np.zeros((NyOut, NxOut))
    if NyIn > 1:
        for kx in range(NxOut):
            InBuffer = Tmp[:, kx]
            OutBuffer = reduce_1D(InBuffer, g, is_centered)
            Out[:, kx] = OutBuffer
    else:
        Out = Tmp.copy()
    return Out

def expand_2D(In, h, is_centered):
    NxIn, NyIn = In.shape
    NxOut = NxIn * 2 if NxIn > 1 else 1
    NyOut = NyIn * 2 if NyIn > 1 else 1
    # X processing
    if NxIn > 1:
        Tmp = np.zeros((NyIn, NxOut))
        for ky in range(NyIn):
            InBuffer = In[ky, :]
            OutBuffer = expand_1D(InBuffer, h, is_centered)
            Tmp[ky, :] = OutBuffer
    else:
        Tmp = In.copy()
    # Y processing
    if NyIn > 1:
        Out = np.zeros((NyOut, NxOut))
        for kx in range(NxOut):
            InBuffer = Tmp[:, kx]
            OutBuffer = expand_1D(InBuffer, h, is_centered)
            Out[:, kx] = OutBuffer
    else:
        Out = Tmp.copy()
    return Out
