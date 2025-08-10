# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 11:33:24 2025

@author: aryan soni
"""

# src/qkernel_utils.py
import numpy as np
import pennylane as qml
from functools import lru_cache

# ---------- devices ----------
def make_device(wires=2, shots=None):
    """Default, noiseless statevector simulator."""
    return qml.device("default.qubit", wires=wires, shots=shots)

# ---------- feature maps ----------
def feature_map(x):
    """Simple 2-feature map: RX on each qubit + CZ entangler."""
    x = np.asarray(x, dtype=float)
    qml.RX(x[0], wires=0)
    qml.RX(x[1], wires=1)
    qml.CZ(wires=[0, 1])

# ---------- kernel element (|<phi(x1)|phi(x2)>|^2) ----------
def make_kernel_element(dev):
    """Returns a function k(x1, x2) -> float using the given device."""

    @qml.qnode(dev)
    def _kernel_probs(x1, x2):
        feature_map(x1)
        qml.adjoint(feature_map)(x2)
        return qml.probs(wires=[0, 1])  # [p00, p01, p10, p11]

    def k(x1, x2) -> float:
        return float(_kernel_probs(x1, x2)[0])  # probability of |00>
    return k

# ---------- kernel matrix ----------
def quantum_kernel_matrix(XA, XB, k_elem):
    """Build K[i,j] = k_elem(XA[i], XB[j])."""
    XA = np.asarray(XA, dtype=float)
    XB = np.asarray(XB, dtype=float)
    K = np.zeros((len(XA), len(XB)))
    for i in range(len(XA)):
        for j in range(len(XB)):
            K[i, j] = k_elem(XA[i], XB[j])
    return K

# Optional cached version (useful when XA/XB repeat)
def make_cached_kernel_element(dev):
    k_raw = make_kernel_element(dev)

    @lru_cache(maxsize=None)
    def _k_cached(x1_tup, x2_tup):
        return k_raw(np.array(x1_tup), np.array(x2_tup))

    def k(x1, x2):
        return _k_cached(tuple(np.asarray(x1, dtype=float)),
                         tuple(np.asarray(x2, dtype=float)))
    return k

# ---------- utilities ----------
def to_angles(Z):
    """Map standardized features to angles in [-pi, pi] column-wise."""
    Z = np.asarray(Z, dtype=float)
    denom = np.maximum(np.max(np.abs(Z), axis=0), 1e-9)
    return np.pi * (Z / denom)
