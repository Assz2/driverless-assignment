"""Candidate solution for the assignment "Map, Data Association & Path Planning".

Run it with::

    python3 solution.py

The script executes the full pipeline step by step and stops at the first
step you have not implemented yet, with a reminder of what is missing.
Implement the three functions below (``associate``, ``correct_map``,
``find_path``), keeping their names and signatures unchanged.
"""

import math
from typing import List, Tuple

from map_generation import Map, Point, generate_map, generate_observations
from visualization import (
    visualize_association_n_correction,
    visualize_map_n_observations,
    visualize_path,
)

# ---------------------------------------------------------------------------
# Parameters of the run.
#
# Feel free to experiment with them, but note that your solution will also be
# evaluated with different values (especially with different random seeds).
# ---------------------------------------------------------------------------
NUM_POINTS = 20
X_RANGE = 20.0
Y_RANGE = 20.0
MAP_SEED = 42

NOISE_STD = 0.25
OBSERVATION_RATE = 0.8
OBSERVATIONS_SEED = 7


# ---------------------------------------------------------------------------
# Provided helper
# ---------------------------------------------------------------------------

def path_length(points: List[Point], path: List[int]) -> float:
    """Total Euclidean length of ``path`` over ``points``.

    Useful to measure how short your path is, e.g. when going for Bonus 1.
    The length is the sum of the Euclidean distances between the consecutive
    nodes of the path.
    """
    return sum(math.dist(points[a], points[b]) for a, b in zip(path, path[1:]))


# ---------------------------------------------------------------------------
# Step 3 -- Data association (yours to implement)
# ---------------------------------------------------------------------------

def associate(map_points: Map, observations: List[Point]) -> List[Tuple[int, int]]:
    """Associate every observation with the map point it was generated from.

    Given the map and the observations (noisy, incomplete and shuffled),
    recover the observation-to-map-point correspondence.

    Returns:
        The association couples as a list of ``(map_index, observation_index)``
        tuples, e.g. ``[(0, 12), (3, 5)]`` means that observation 12 was
        generated from map point 0, and observation 5 from map point 3.
    """
    raise NotImplementedError("Step 3 (data association) is yours to implement")


# ---------------------------------------------------------------------------
# Step 4 -- Map correction (yours to implement)
# ---------------------------------------------------------------------------

def correct_map(
    map_points: Map,
    observations: List[Point],
    associations: List[Tuple[int, int]],
) -> Map:
    """Correct the map, given the data association.

    For every association couple ``(i, j)``, the corrected position of map
    point ``i`` is the average (midpoint) of ``map_points[i]`` and
    ``observations[j]``. Map points without an associated observation keep
    their original position.

    Returns:
        The corrected map: a list of ``(x, y)`` tuples with the same length
        and the same order as ``map_points``.
    """
    raise NotImplementedError("Step 4 (map correction) is yours to implement")


# ---------------------------------------------------------------------------
# Step 5 -- Path finding (yours to implement)
# ---------------------------------------------------------------------------

def find_path(corrected_map: Map) -> List[int]:
    """Find a path that visits every node of the corrected map exactly once.

    Return the path as a list of node indices in visit order, e.g.
    ``[4, 0, 11, 2, ...]``. A path that includes each node exactly once is
    usually called a *Hamiltonian path*: any permutation of the node indices
    is a formally valid answer, but:

    * **Bonus 1 (+1 point)**: use a heuristic or an algorithm (e.g. nearest
      neighbour, greedy insertion, 2-opt, ...) so that the total length of
      your path is clearly shorter than the worst possible path. You can
      measure it with :func:`path_length`.

    * **Bonus 2 (+1 point)**: no arrow of the path goes over a previous
      arrow (arrows that only touch at a shared node do not count as
      crossings). You can check it visually with ``visualize_path``.
    """
    raise NotImplementedError("Step 5 (path finding) is yours to implement")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    # Step 1 -- Generate the map (provided) ---------------------------------
    map_points = generate_map(
        num_points=NUM_POINTS,
        x_range=X_RANGE,
        y_range=Y_RANGE,
        seed=MAP_SEED,
    )

    # Step 2 -- Generate the observations (provided) ------------------------
    observations = generate_observations(
        map_points,
        noise_std=NOISE_STD,
        observation_rate=OBSERVATION_RATE,
        seed=OBSERVATIONS_SEED,
    )
    visualize_map_n_observations(map_points, observations, X_RANGE, Y_RANGE)

    # Step 3 -- Data association (yours) -------------------------------------
    try:
        associations = associate(map_points, observations)
    except NotImplementedError:
        print("Step 3 is not implemented yet: fill in associate() in solution.py.")
        return

    # Step 4 -- Map correction (yours) ---------------------------------------
    try:
        corrected_map = correct_map(map_points, observations, associations)
    except NotImplementedError:
        print("Step 4 is not implemented yet: fill in correct_map() in solution.py.")
        return
    visualize_association_n_correction(
        map_points,
        observations,
        corrected_map,
        associations,
        X_RANGE,
        Y_RANGE,
    )

    # Step 5 -- Path finding (yours) -----------------------------------------
    try:
        path = find_path(corrected_map)
    except NotImplementedError:
        print("Step 5 is not implemented yet: fill in find_path() in solution.py.")
        return
    visualize_path(corrected_map, path)

    print(
        f"Pipeline complete: {len(associations)} associations over "
        f"{len(observations)} observations, path over {len(corrected_map)} "
        f"nodes with total length {path_length(corrected_map, path):.2f}."
    )


if __name__ == "__main__":
    main()
