Installation and Getting Started
================================

Installation
------------
The best way to install Rogers is from PyPI. To run on GPU (the intended use case), you must specify the CUDA version in use; currently, CUDA 12 and CUDA 13 are supported. To install for CUDA 13, run::

    pip install rogers-neighbor[cuda13]

Basic Use
---------
There are two main data structures in use in Rogers. The primary one is :class:`.NeighborCalculator`, which performs all the calculations and holds all the necessary state information (i.e., array sizes). The second is :class:`.NeighborList`, which is simply a Python ``NamedTuple`` dataclass that holds a neighbor list.

.. note::
   While Rogers is written in Jax for speed, it is not compatible with Jax transformations such as ``jit`` and ``vmap``. This is because the size of the neighbor list cannot be known at compile time.

To use Rogers, you must first import the package and initialize a calculator:

>>> import rogers
>>> nl_calc = rogers.NeighborCalculator()

To compute a neighbor list, you must have:

* An N×3 array of point positions, given in Cartesian coordinates.
* A floating-point cutoff radius.
* If using periodic boundary conditions, the cell vectors of a unit cell.
* If your position array is padded, an array indicating which particles are real and which are for padding.

Numpy or Jax arrays work for input, but output will always be given as a Jax array. Note that, with periodic boundary conditions, the cutoff must be less than half the shortest cell dimension.

>>> import numpy as np
>>> pos = np.array([[0.5, -1.2, 3.6],
...                 [2.2, 4.7, -5.2],
...                 [0.6, -1.2, 4.1],
...                 [0., 0., 0.]])
>>> cutoff = 3.
>>> valid = np.array([True, True, True, False])
>>> cell = 7.*np.eye(3)

Now we are ready to build a neighbor list. Operation is simple:

>>> nlist = nl_calc.build_neighbor_list(
...     pos, cutoff, cell=cell, valid=valid)
>>> nlist
NeighborList(first=Array([ 1,  0, -1, ..., -1, -1, -1], dtype=int32), second=Array([ 0,  2, -1, ..., -1, -1, -1], dtype=int32))
>>> nlist.first.size
4096
>>> nlist.first[:5]
Array([ 1,  0, -1, -1, -1], dtype=int32)
>>> nlist.second[:5]
Array([ 0,  2, -1, -1, -1], dtype=int32)

The convention is that, for any ``i``, ``(nlist.first[i], nlist.second[i])`` is a neighboring pair. Any entry with a ``-1`` is considered an invalid pair and should be ignored by any downstream users. A few things to notice:

* The size of the neighbor list is 4096. This is the default minimum size, which may increase as needed.
* The point ``0`` is a neighbor of both other points; its closeness to ``2`` is apparent, and the closeness to ``1`` happens because of the periodic wrapping.

We have constructed our very first neighbor list. Others can be constructed in much the same way.
