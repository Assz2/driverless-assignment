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
    matrix_distances : List[List[float]] = []
    absolute_min : float = float('inf')
    absolute_max : float = float('-inf')
    already_associated_index : dict[int, float] = {}
    result : List[Tuple[int, int]] = []
    correction_cols_indexes : dict[int, List[int]] = {}

    # create distance matrix -> row is map_index, col is obs_index
    for map_point in map_points:
        row : List[float] = []
        for obs_index, obs_point in enumerate(observations):
            distance = math.dist(map_point, obs_point)

            # search for absolute min and max distances for creating an artificial data-based threshold to categorize non associated points
            if distance < absolute_min: absolute_min = distance
            if distance > absolute_max: absolute_max = distance
            row.append(distance)
        matrix_distances.append(row)
    threshold_distance = (absolute_max - absolute_min) / 10


    for row_idx, row in enumerate(matrix_distances):
        min_distance = float('inf')
        obs_index = -1

        for col, distance in enumerate(row):
            # if the distance is greater than the threshold, skip iteration
            if distance > threshold_distance:
                continue

            if col in already_associated_index:
                # if the observation has been already used and the current distance is greater than the stored one, skip iteration
                if distance >= already_associated_index[col]:
                    continue

                # else the observation has been already used and the current distance is less than the stored one 
                # -> this observation better suits the current map point, change the association
                else:
                    map_idx_to_fix = -1
                    for i, (map_idx, obs_idx) in enumerate(result):
                        if obs_idx == col:
                            map_idx_to_fix = result[i][0]
                            break

                    if map_idx_to_fix not in correction_cols_indexes:
                        correction_cols_indexes[map_idx_to_fix] = []
                    correction_cols_indexes[map_idx_to_fix].append(col)

                    _, new_obs_idx = min( (distance, j) for j, distance in enumerate(matrix_distances[map_idx_to_fix]) if j not in correction_cols_indexes.get(map_idx_to_fix, []))
                    result[i] = (map_idx, new_obs_idx)
                         
            if distance < min_distance:
                min_distance = distance
                obs_index = col

        # if the map point is associated with one observation, save it to the result list
        if min_distance != float('inf'):
            result.append((row_idx, obs_index))
            already_associated_index[obs_index] = min_distance


    return result

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
    
    corrected_map : Map = []

    associated_indices = {map_index for map_index, obs_index in associations}
    missing_indices = [ ind for ind, _ in enumerate(map_points) if ind not in associated_indices ]
    associated_dict = dict(associations)

    for map_index, _ in enumerate(map_points):
        if map_index in missing_indices:
            corrected_map.append(map_points[map_index])
            continue

        obs_index = associated_dict.get(map_index)
        map_coord = map_points[map_index]
        obs_coord = observations[obs_index]
        corrected_coord = (map_coord[0] + obs_coord[0]) / 2, (map_coord[1] + obs_coord[1]) / 2
        corrected_map.append(corrected_coord)
    return corrected_map


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

    visit_order_indexes : List[int] = []
    # given a node in the corrected map, find the nearest neighbour node that has not been visited yet
    for index, coord in enumerate(corrected_map):
        i = index + 1
        nn_index = -1
        min_distance = float('inf')
        while i < len(corrected_map):
            nn_distance = math.dist(coord, corrected_map[i])
            if nn_distance < min_distance:
                print("Better distance found: ", {nn_distance})
                min_distance = nn_distance
                nn_index = i
            i += 1
        # it's the last index
        if nn_index == -1:
            nn_index = index
        visit_order_indexes.append(nn_index)

    return visit_order_indexes


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
