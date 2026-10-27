import itertools
import os
import sys
from collections import Counter

import numpy as np
import pytest

import rogers as neighbors
from rogers import NO_CELL

# -- neighbor list vs brute-force reference --

def brute_force_pairs(cell, cutoff, positions, single_count=True, n_images=3):
    """Naive reference: explicitly try all atom pairs across +/- n_images
    periodic replications of the cell and keep those within cutoff."""
    na = positions.shape[1]
    shifts = itertools.product(range(-n_images, n_images+1), repeat=3)
    shifts = [cell @ np.array(s, dtype=float) for s in shifts]

    pairs = []
    for i in range(na):
        for j in range(na):
            if single_count and j > i:
                continue
            for shift in shifts:
                dx = positions[:,j] + shift - positions[:,i]
                r2 = dx @ dx
                if (r2 <= cutoff**2) and not (i == j and r2 == 0.0):
                    pairs.append((i,j))
    return Counter(pairs)

# non-orthorhombic reference cell shared by the neighbor-list tests below
_TRICLINIC_CELL = np.array([[3.0, 0.5, 0.0],
                            [0.0, 2.7, 0.3],
                            [0.0, 0.0, 3.3]])
_CUTOFF = 2.5 # > half cell-width so we will have multiple images


def _raw_pair_counter(result):
    """`build_neighbor_list` pads its output with (-1, -1) up to
    `_max_pairs_cap`; strip that before comparing."""
    i_arr, j_arr = np.asarray(result[0]), np.asarray(result[1])
    valid = (i_arr >= 0) & (j_arr >= 0)
    return Counter(zip(i_arr[valid].tolist(), j_arr[valid].tolist()))


def canonicalize_counter(counter):
    """Fold each pair onto a canonical (min, max) key."""
    folded = Counter()
    for (i, j), n in counter.items():
        folded[(min(i, j), max(i, j))] += n
    return folded


@pytest.mark.parametrize("single_count", [True, False])
@pytest.mark.parametrize("use_pbc", [True, False])
def test_neighbor_list(single_count, use_pbc):
    """Test built-in neighbor list method against brute force algorithm, for
    both half/full lists (`single_count`) and both PBC and non-PBC
    (`NO_CELL`) modes."""
    rng = np.random.default_rng(0)
    positions = _TRICLINIC_CELL @ rng.uniform(0.0, 1.0, size=(3, 8))
    # NO_CELL disables PBC: build_neighbor_list pads its own large box
    # internally, so the brute-force reference should see no periodic images.
    cell = _TRICLINIC_CELL if use_pbc else NO_CELL
    n_images = 3 if use_pbc else 0

    nl_calc = neighbors.NeighborCalculator()
    result = nl_calc.build_neighbor_list(
        positions.T,
        _CUTOFF,
        cell=cell,
        single_count=single_count,
    )
    result = (result.first, result.second)
    got = canonicalize_counter(_raw_pair_counter(result))

    expected = canonicalize_counter(brute_force_pairs(
        _TRICLINIC_CELL, _CUTOFF, positions,
        single_count=single_count, n_images=n_images))
    assert got == expected

def test_neighbor_list_padded_mol():
    rng = np.random.default_rng(0)
    positions = _TRICLINIC_CELL @ rng.uniform(0.0, 1.0, size=(3,8))
    cell = _TRICLINIC_CELL
    valid = np.array([1,1,1,1,1,0,0,0], dtype=bool)
    nl_calc = neighbors.NeighborCalculator()
    result = nl_calc.build_neighbor_list(
            positions.T, _CUTOFF, cell=cell, valid=valid)
    result = (result.first, result.second)
    got = canonicalize_counter(_raw_pair_counter(result))
    expected = canonicalize_counter(
        brute_force_pairs(_TRICLINIC_CELL, _CUTOFF, positions[:,:-3]))
    assert got == expected
    

def test_neighbor_list_padding_after_larger_system():
    """`max_pairs_cap` only grows across calls on the same calculator, so a
    smaller system's output gets padded to a larger cap left over from a
    prior, bigger call. Lock in that the padding is stripped correctly and
    doesn't corrupt the (still-correct) real pairs."""
    rng = np.random.default_rng(0)
    nl_calc = neighbors.NeighborCalculator()

    positions_16 = _TRICLINIC_CELL @ rng.uniform(0.0, 1.0, size=(3, 16))
    nl_calc.build_neighbor_list(
        positions_16.T, _CUTOFF, cell=_TRICLINIC_CELL
    )
    cap_after_large = nl_calc.max_pairs_cap

    positions_8 = _TRICLINIC_CELL @ rng.uniform(0.0, 1.0, size=(3, 8))
    result = nl_calc.build_neighbor_list(
        positions_8.T, _CUTOFF, cell=_TRICLINIC_CELL
    )
    i_arr = np.asarray(result.first)
    assert i_arr.shape[0] == cap_after_large # padded up to the leftover cap
    result = (result.first, result.second)
    got = canonicalize_counter(_raw_pair_counter(result))

    expected = canonicalize_counter(brute_force_pairs(
        _TRICLINIC_CELL, _CUTOFF, positions_8, single_count=True, n_images=3))
    assert got == expected

