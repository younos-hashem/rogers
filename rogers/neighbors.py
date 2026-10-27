from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

NO_CELL = np.zeros((3,3), dtype=jnp.float32)

###################
## Helper functions
###################
def estimate_ncr(n: int, vol: float, cutoff: float) -> int:
    r"""Estimate the optimal value of NCR, the ratio of cutoff to mesh cell size.
    Based on an empirical finding that performance (on CPU) is optimized when
    the mean number of atoms per cell is ~8.
    """
    return max(1, min( round((n/vol/8)**(1/3)*cutoff), 5 ))


##################
## Data structures
##################
@jax.tree_util.register_dataclass
@dataclass
class NeighborList:
    r"""Holds a neighbor list in which `(first[i], second[i])` is a neighboring
    pair. `-1` is used as a padding sentinel.
    """
    first: Int[Array, " n_neighb"]
    second: Int[Array, " n_neighb"]

class NeighborCalculator:
    r"""Calculates neighbor lists for a set of points in 3D space."""
    def __init__(self) -> None:
        self.max_count_cap = 8
        self.single_pair_cap = 64
        self.max_pairs_cap = 4096

    def build_neighbor_list(
            self,
            pos: Float[Array, " n_tot 3"],
            cutoff: float,
            cell: Float[Array, "3 3"] = None,
            valid: Bool[Array, " n_tot"] = None,
            single_count: bool = True,
    ) -> NeighborList:
        r"""Compute a neighbor list for a single molecule.

        Args:
            pos: (n_tot, 3) Array of positions of all points in 3D Cartesian
                coordinates.
            cutoff: Cutoff radius within which to consider two particles
                neighbors. If using periodic boundary conditions, must be less
                than half the box size.
            cell: (3, 3) Array of unit cell box vectors, one per column. If not
                using periodic boundary conditions, pass `None`.
            valid: (n_tot,) Boolean array indicating whether to include
                (`True`) each point in the neighbor search. This is intended
                to be used with padded arrays; if `None`, all points are
                assumed valid.
            single_count: If `True` (default), only returns pairs with i <= j

        Returns:
            :py:class`NeighborList`
        """
        if valid is None:
            valid = np.ones(pos.shape[0], dtype=bool)
        if cell is None:
            cell = NO_CELL
        periodic = True

        # Build cell for non-PBC
        if np.array_equal(cell, NO_CELL):
            mins = np.min(pos, axis=0)
            maxes = np.max(pos, axis=0)
            size = (maxes - mins)*1.02 + 1  # safety margin
            cell = np.diag(size)
            pos = pos - mins[None,:]
            periodic = False

        V = np.abs(np.linalg.det(cell))
        n = pos.shape[0]
        ncr = estimate_ncr(n, V, cutoff)

        rcell = np.linalg.inv(cell.T)
        nmxv = np.maximum(np.ones(3, dtype=np.int32),
                        (0.999 * ncr / cutoff /
                            np.linalg.norm(rcell, axis=0)).astype(np.int32))
        celmsh = cell / nmxv[None, :]
        rmcell = rcell * nmxv[None, :]
        nnxv = 1 + np.astype(cutoff * np.linalg.norm(rmcell, axis=0),
                              np.int32)
        n_mesh_div, n_neighb_cells = tuple(nmxv.tolist()), tuple(nnxv.tolist())
        nnx1, nnx2, nnx3 = n_neighb_cells
        shift_grid = np.mgrid[-nnx1:nnx1+1, -nnx2:nnx2+1, -nnx3:nnx3+1]
        cell_shifts = np.einsum('ij,jklm->klmi', celmsh, shift_grid).reshape(
                -1, 3)
        all_idx = np.arange(n)

        # now run everything on GPU at least once until we stop overflowing
        need_bin = True
        while True:
            if need_bin:
                (max_count, in_mesh_pos_sorted, mesh_idx_sorted, valid_sorted,
                cells, order) = (
                    bin_points(
                        n_mesh_div, pos, cell, celmsh, valid,
                        self.max_count_cap)
                    )
            nnas, jans = find_all(
                all_idx, cutoff, in_mesh_pos_sorted, celmsh, mesh_idx_sorted,
                shift_grid, cell_shifts, cells, n_mesh_div, valid_sorted,
                periodic, single_count, self.single_pair_cap
            )
            n_pairs, i_final, j_final = compact(
                nnas, jans, order, self.max_pairs_cap
            )
            max_nna = jnp.max(nnas)
            max_count, max_nna, n_pairs = jax.device_get((
                max_count,max_nna,n_pairs))
            if max_count > self.max_count_cap:
                self.max_count_cap = -(-max_count // 8)*8
                continue
            grew = False
            if max_nna > self.single_pair_cap:
                grew = True
                self.single_pair_cap *= 2
            if (n_pairs > self.max_pairs_cap):
                grew = True
                self.max_pairs_cap *= 2
            if not grew:
                break
            need_bin = False

        return NeighborList(i_final, j_final)

#######################
# Calculation functions   
#######################

@jax.jit(static_argnames=["n_mesh_div", "max_count_cap"])
def bin_points(
        n_mesh_div: tuple[int, int, int],
        pos: Float[Array, " n 3"],
        cell: Float[Array, "3 3"],
        celmsh: Float[Array, "3 3"],
        valid: Bool[Array, " n"],
        max_count_cap: int,
) -> tuple[int,
           Float[Array, " n 3"],
           Int[Array, " nm max_count_cap"]]:
    r"""Bin the points into their respective mesh-cells.

    Args:
        n_mesh_div: (3,) Number of mesh divisions in each dimension
        n_neighb_cell: (3,) Number of neighboring mesh cells in each dimension
        pos: (n, 3) Positions of all points
        cell: (3, 3) Unit cell vectors, one per column
        valid: (n,) Whether each point is valid or padded
        max_count_cap: Static; maximum number of points per cell

    Returns:
        * (int) Actual max number of points per cell, to check overflows
        * (n, 3) In-mesh position of each point
        * (n, 3) Mesh-cell index of each point
        * (n,) Which atoms are valid with the new sorting
        * (nm, max_count_cap) Indices of points in each cell, sorted for
            locality
        * (n,) Mapping of new indices to old indices
    """
    nmxv = jnp.array(n_mesh_div)
    nm = int(np.prod(n_mesh_div))
    n = pos.shape[0]
    rcell = jnp.linalg.inv(cell.T)
    rmcell = rcell * nmxv[None, :]

    # Find which mesh cell a point is in
    in_mesh_relpos = pos @ rmcell  # (n, 3) in-cell position in mesh coords.
    mesh_cell_idx = jnp.floor(in_mesh_relpos).astype(jnp.int32)
    in_mesh_relpos = in_mesh_relpos - mesh_cell_idx
    mesh_cell_idx = mesh_cell_idx % nmxv[None, :]
    flat_mesh_idx = jnp.ravel_multi_index(mesh_cell_idx.T, n_mesh_div,
                                          mode="ignore")
    in_mesh_pos = in_mesh_relpos @ celmsh.T

    eff_mesh_idx = jnp.where(valid, flat_mesh_idx, nm)  # invalid -> fake cell
    pts_by_cell = jnp.argsort(eff_mesh_idx, stable=False)
    flat_sorted = eff_mesh_idx[pts_by_cell]
    mesh_idx_sorted = mesh_cell_idx[pts_by_cell]  # sorted list of occ. cells
    counts = jnp.bincount(eff_mesh_idx, length=nm+1)  # pts per cell
    max_count = counts[:nm].max() if n > 0 else 1
    starts = jnp.cumsum(counts) - counts  # pointers to cells
    rank = jnp.arange(n) - starts[flat_sorted]  # pt. rank w/in cell

    # reorder points so those in the same cell are next to each other, which
    # helps with memory locality
    order = pts_by_cell
    in_mesh_pos_sorted = in_mesh_pos[order, :]
    valid_sorted = valid[order]

    # sorted idx k goes in cell flat_sorted[k] at rank rank[k]
    idx = jnp.arange(n, dtype=jnp.int32)
    cells = jnp.full((nm, max_count_cap), -1, dtype=jnp.int32)
    cells = cells.at[jnp.where(valid_sorted, flat_sorted, nm), rank].set(
        idx, mode="drop"
    )

    return (
        max_count, in_mesh_pos_sorted, mesh_idx_sorted, valid_sorted, cells,
        order
    )


def find_neighbors_single(
        i0: int,
        cutoff: float,
        in_mesh_pos: Float[Array, " n 3"],
        celmsh: Float[Array, "3 3"],
        mesh_idx: Int[Array, " n 3"],
        shift_grid: Int[Array, "3 2*nnx+1"],
        cell_shifts: Float[Array, " nnm 3"],
        cells: Int[Array, " nm max_count_cap"],
        n_mesh_div: tuple[int, int, int],
        valid: Bool[Array, " n"],
        periodic: bool,
        single_count: bool,
        cap: int,
) -> tuple[bool, int, Int[Array, " nna"]]:
    r"""Find all neighbors of a given point within a certain radius.

    Args:
        i0: Index of point whose neighbors are to be found
        cutoff: Cutoff within which to consider points neighbors
        in_mesh_pos: (n,3) In–mesh-cell position of each point
        celmsh: (3,3) Mesh cell lattice vectors, one per column
        mesh_idx: (n,3) Home–mesh-cell index of each point
        shift_grid: (3,2*nnx+1) Grid of # of shifts to get to each neighboring
            mesh cell
        cell_shifts: (nnm, 3) Vector shift to all neighboring cells
        cells: (nm, max_count_cap) Indices of all points in each cell
        n_mesh_div: Number of mesh divisions in each direction
        valid: Which points are real/valid (vs. padding)
        periodic: Whether to apply periodic wrapping
        single_count: Whether to give a half-list (True) or a full-list
        cap: Maximum number of neighbors
    
    Returns:
        * (int) number of actual neighbors (vs. padding)
        * (cap,) Indices of neighbors, padded with `-1`
    """
    range2 = cutoff**2
    nmxv = jnp.array(n_mesh_div)

    home_cell = mesh_idx[i0]
    neighb_cells = shift_grid + home_cell[:, None, None, None]
    per_shift, neighb_cells = jnp.divmod(neighb_cells, nmxv[:,None,None,None])
    shifted = (per_shift != 0).any(axis=0).reshape(-1)  # (nnm,)
    neighb_cells_1d = jnp.ravel_multi_index(neighb_cells, n_mesh_div,
                                            mode='ignore').reshape(-1)
    pts_in_neighbs = cells[neighb_cells_1d]  # (nnm, max_count_cap)
    disp = (
        cell_shifts[:, None, :]  # (nnm, 1, 3)
        + in_mesh_pos[pts_in_neighbs]  # (nnm, max_count_cap, 3)
        - in_mesh_pos[i0][None, None, :]  # (1, 1, 3)
    )
    r2 = jnp.sum(jnp.square(disp), axis=2)  # (nnm, max_count_cap)
    mask = pts_in_neighbs > -1
    within = r2 < range2
    not_trivial_self = ~((r2 == 0.0) & (pts_in_neighbs == i0))
    half_list = jnp.logical_not(single_count) | (i0 <= pts_in_neighbs)
    pbc_or_noshift = (periodic | ~shifted)[:, None]
    include = (
        mask & within & not_trivial_self & half_list & pbc_or_noshift
        & valid[i0]
    )
    keep = include.ravel()
    starts = jnp.cumsum(keep) - 1
    nna = starts[-1] + 1
    out = jnp.full(cap, -1).at[jnp.where(keep, starts, cap)].set(
        pts_in_neighbs.ravel(), mode='drop'
    )
    return nna, out

@jax.jit(static_argnames=["max_pairs_cap"])
def compact(
        nnas: Int[Array, " n"],
        jans: Int[Array, " n single_pair_cap"],
        order: Int[Array, " n"],
        max_pairs_cap: int
) -> tuple[int, Int[Array, " max_pairs_cap"], Int[Array, " max_pairs_cap"]]:
    n = nnas.size
    single_cap = jans.shape[1]
    offsets = jnp.cumsum(nnas) - nnas  # (n,); where each starts
    rank = jnp.arange(single_cap)[None,:]  # (1,single_cap) rank for parent pt.
    real = rank < nnas[:, None]  # (n, single_cap)
    # slots: (n, single_cap) where j goes in final list
    slots = jnp.where(real, rank + offsets[:,None], max_pairs_cap)
    empty = jnp.full(max_pairs_cap, -1, dtype=jnp.int32)
    i = jnp.broadcast_to(order[:, None], jans.shape) # (n, single_cap)
    j = order[jnp.clip(jans, 0, n-1)]
    i_final = empty.at[slots].set(i, mode='drop')
    j_final = empty.at[slots].set(j, mode='drop')
    return (jnp.sum(nnas), i_final, j_final)

find_all = jax.jit(jax.vmap(find_neighbors_single, in_axes=(0,) + (None,)*12),
                   static_argnames=["single_count", "cap", "n_mesh_div"])
