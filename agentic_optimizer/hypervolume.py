"""Hypervolume indicator computation.

Based on the Fonseca et al. dimension-sweep algorithm:
C. M. Fonseca, L. Paquete, and M. Lopez-Ibanez. "An improved dimension-sweep
algorithm for the hypervolume indicator." IEEE CEC, 2006.
"""

from __future__ import annotations
from typing import List

import numpy as np

from agentic_optimizer.individual import Individual


def hypervolume(elite: List[Individual], ref) -> float:
    """Compute hypervolume dominated by elite set w.r.t. reference point."""
    pointset = [tuple(e.objs) for e in elite]
    ref = np.array(ref, dtype=float)
    if len(ref) == 2:
        return _hypervolume_2d(pointset, ref)
    hv = _HyperVolume(ref)
    return hv.compute(pointset)


def hypervolume_from_points(points: np.ndarray, ref: np.ndarray) -> float:
    """Compute hypervolume from a numpy array of points and reference."""
    pointset = [tuple(p) for p in points]
    ref = np.asarray(ref, dtype=float)
    if len(ref) == 2:
        return _hypervolume_2d(pointset, ref)
    hv = _HyperVolume(ref)
    return hv.compute(pointset)


def _hypervolume_2d(pointset, ref) -> float:
    """O(n log n) hypervolume for 2 objectives via sorted sweep."""
    pts = np.array(pointset, dtype=float)
    mask = (pts[:, 0] < ref[0]) & (pts[:, 1] < ref[1])
    pts = pts[mask]
    if len(pts) == 0:
        return 0.0
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    nd = [pts[0]]
    min_f2 = pts[0, 1]
    for i in range(1, len(pts)):
        if pts[i, 1] < min_f2:
            nd.append(pts[i])
            min_f2 = pts[i, 1]
    nd = np.array(nd)
    n = len(nd)
    f1_right = np.empty(n)
    f1_right[:-1] = nd[1:, 0]
    f1_right[-1] = ref[0]
    widths = f1_right - nd[:, 0]
    heights = ref[1] - nd[:, 1]
    return float(np.sum(widths * heights))


class _HyperVolume:
    """Hypervolume computation (Fonseca variant 3). Assumes minimization."""

    def __init__(self, referencePoint):
        self.referencePoint = referencePoint
        self.list = []

    def compute(self, front):
        relevantPoints = np.array(front, dtype=float)
        referencePoint = self.referencePoint
        dimensions = len(referencePoint)
        if any(referencePoint):
            relevantPoints = relevantPoints - referencePoint
        self.preProcess(relevantPoints)
        bounds = [-1.0e308] * dimensions
        return self.hvRecursive(dimensions - 1, len(relevantPoints), bounds)

    def hvRecursive(self, dimIndex, length, bounds):
        hvol = 0.0
        sentinel = self.list.sentinel
        if length == 0:
            return hvol
        elif dimIndex == 0:
            return -sentinel.next[0].cargo[0]
        elif dimIndex == 1:
            q = sentinel.next[1]
            h = q.cargo[0]
            p = q.next[1]
            while p is not sentinel:
                pCargo = p.cargo
                hvol += h * (q.cargo[1] - pCargo[1])
                if pCargo[0] < h:
                    h = pCargo[0]
                q = p
                p = q.next[1]
            hvol += h * q.cargo[1]
            return hvol
        else:
            remove = self.list.remove
            reinsert = self.list.reinsert
            hvRecursive = self.hvRecursive
            p = sentinel
            q = p.prev[dimIndex]
            while q.cargo is not None:
                if q.ignore < dimIndex:
                    q.ignore = 0
                q = q.prev[dimIndex]
            q = p.prev[dimIndex]
            while length > 1 and (q.cargo[dimIndex] > bounds[dimIndex] or q.prev[dimIndex].cargo[dimIndex] >= bounds[dimIndex]):
                p = q
                remove(p, dimIndex, bounds)
                q = p.prev[dimIndex]
                length -= 1
            qArea = q.area
            qCargo = q.cargo
            qPrevDimIndex = q.prev[dimIndex]
            if length > 1:
                hvol = qPrevDimIndex.volume[dimIndex] + qPrevDimIndex.area[dimIndex] * (qCargo[dimIndex] - qPrevDimIndex.cargo[dimIndex])
            else:
                qArea[0] = 1
                qArea[1:dimIndex+1] = [qArea[i] * -qCargo[i] for i in range(dimIndex)]
            q.volume[dimIndex] = hvol
            if q.ignore >= dimIndex:
                qArea[dimIndex] = qPrevDimIndex.area[dimIndex]
            else:
                qArea[dimIndex] = hvRecursive(dimIndex - 1, length, bounds)
                if qArea[dimIndex] <= qPrevDimIndex.area[dimIndex]:
                    q.ignore = dimIndex
            while p is not sentinel:
                pCargoDimIndex = p.cargo[dimIndex]
                hvol += q.area[dimIndex] * (pCargoDimIndex - q.cargo[dimIndex])
                bounds[dimIndex] = pCargoDimIndex
                reinsert(p, dimIndex, bounds)
                length += 1
                q = p
                p = p.next[dimIndex]
                q.volume[dimIndex] = hvol
                if q.ignore >= dimIndex:
                    q.area[dimIndex] = q.prev[dimIndex].area[dimIndex]
                else:
                    q.area[dimIndex] = hvRecursive(dimIndex - 1, length, bounds)
                    if q.area[dimIndex] <= q.prev[dimIndex].area[dimIndex]:
                        q.ignore = dimIndex
            hvol -= q.area[dimIndex] * q.cargo[dimIndex]
            return hvol

    def preProcess(self, front):
        dimensions = len(self.referencePoint)
        nodeList = _MultiList(dimensions)
        nodes = [_MultiList.Node(dimensions, point) for point in front]
        for i in range(dimensions):
            self.sortByDimension(nodes, i)
            nodeList.extend(nodes, i)
        self.list = nodeList

    def sortByDimension(self, nodes, i):
        decorated = [(node.cargo[i], idx, node) for idx, node in enumerate(nodes)]
        decorated.sort(key=lambda x: x[0])
        nodes[:] = [node for (_, _, node) in decorated]


class _MultiList:
    class Node:
        def __init__(self, numberLists, cargo=None):
            self.cargo = cargo
            self.next = [None] * numberLists
            self.prev = [None] * numberLists
            self.ignore = 0
            self.area = [0.0] * numberLists
            self.volume = [0.0] * numberLists

    def __init__(self, numberLists):
        self.numberLists = numberLists
        self.sentinel = _MultiList.Node(numberLists)
        self.sentinel.next = [self.sentinel] * numberLists
        self.sentinel.prev = [self.sentinel] * numberLists

    def extend(self, nodes, index):
        sentinel = self.sentinel
        for node in nodes:
            lastButOne = sentinel.prev[index]
            node.next[index] = sentinel
            node.prev[index] = lastButOne
            sentinel.prev[index] = node
            lastButOne.next[index] = node

    def remove(self, node, index, bounds):
        for i in range(index):
            predecessor = node.prev[i]
            successor = node.next[i]
            predecessor.next[i] = successor
            successor.prev[i] = predecessor
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]
        return node

    def reinsert(self, node, index, bounds):
        for i in range(index):
            node.prev[i].next[i] = node
            node.next[i].prev[i] = node
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]
