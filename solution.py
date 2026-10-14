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
    absolute_min = float('inf')
    absolute_max = float('-inf')
    correction_cols_indexes : dict[int, List[int]] = {}
    already_associated_index : dict[int, float] = {}
    result : List[Tuple[int, int]] = []

    # iterations to create matrix of distances -> row is map_index, col is obs_index
    for map_point in map_points:
        row : List[float] = []
        for obs_point in observations:
            distance = math.dist(map_point, obs_point)

            # search for absolute min and max distances for creating an artificial data-based threshold to categorize non associated points
            if distance < absolute_min: absolute_min = distance
            if distance > absolute_max: absolute_max = distance

            row.append(distance)
        matrix_distances.append(row)
    # artificial data-based threshold
    threshold_distance = (absolute_max - absolute_min) / 10

    # iterations to associate data points
    for row_idx, row in enumerate(matrix_distances):
        min_distance = float('inf')
        obs_index = -1

        for col_idx, distance in enumerate(row):
            # if the distance is greater than the threshold, skip iteration
            if distance > threshold_distance:
                continue

            if col_idx in already_associated_index:
                # if the observation has been already used and the current distance is greater than the already associated one, 
                # skip iteration
                if distance >= already_associated_index[col_idx]: continue

                # if the observation has been already used 
                # but the current distance is less than the distance of the already associated map_points - observation 
                # -> this observation better suits the current map point, change the association
                else:
                    # we need to backtrack -> extract related map_point, reassociate it with another observation, 
                    # append this observation to the correct map_point
                    backtracking_map_idx = -1
                    i = 0

                    # here I want to find the map_idx to which change observation. I am extracting the map_idx from result collection
                    # algorithm invariant -> ater this loop, backtracking_map_idx should never be -1 
                    for _, obs_idx_result in result:
                        if obs_idx_result == col_idx:
                            backtracking_map_idx = result[i][0]
                            break
                        i += 1

                    # correction_cols_indexes is a dictionary: 
                    # key - map_idx to backtrack, values - list of nodes that better suits the actua map_point 
                    if backtracking_map_idx not in correction_cols_indexes: correction_cols_indexes[backtracking_map_idx] = []
                    correction_cols_indexes[backtracking_map_idx].append(col_idx)

                    # new column (observation) index to associate for backtracked map_point (row corresponding to matrix_distances[backtracking_map_idx]) 
                    # -> result collection is edited
                    new_obs_idx, dist = min( (j, distance) for j, distance in enumerate(matrix_distances[backtracking_map_idx]) if j not in correction_cols_indexes.get(backtracking_map_idx, []))
                    already_associated_index[new_obs_idx] = dist
                    result[i] = (backtracking_map_idx, new_obs_idx)
                         
            elif distance < min_distance:
                min_distance = distance
                obs_index = col_idx

        # if the map point is associated with one observation, save it to the result list
        if obs_index != -1:
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

    # set of map_points idxs observed (correctly associated)
    associated_indices = {map_index for map_index, _ in associations}

    # set of indxs of map_point without association
    missing_indices = { ind for ind, _ in enumerate(map_points) if ind not in associated_indices }

    # key - map_indxs, value - obs_indxs
    associated_dict = dict(associations)

    for map_index, _ in enumerate(map_points):
        if map_index in missing_indices:
            corrected_map.append(map_points[map_index])
            continue

        obs_index = associated_dict.get(map_index)
        map_coord : Tuple[int, int] = map_points[map_index]
        obs_coord : Tuple[int, int] = observations[obs_index]
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
    matrix_distances : List[List[float]] = []
    visit_order_indexes : List[int] = []

    for point in corrected_map:
        row : List[Point] = []
        for target_point in corrected_map:
            distance = math.dist(point, target_point)
            # this checks if row_idx == col_idx -> move not allowed, set with high distance
            if distance == 0:  distance = float('inf') 
            row.append(distance)
        matrix_distances.append(row)

    already_visited : set = set()
    position_idx = 0
    already_visited.add(position_idx)
    visit_order_indexes.append(position_idx)
    
    while len(already_visited) < len(corrected_map):
        min_dist : float = float('inf')
        save_idx = -1
        point = corrected_map[position_idx]

        for next_step_idx, next_step_point in enumerate(corrected_map):
            if next_step_idx in already_visited or next_step_idx == position_idx:
                continue

            dist = math.dist(point, next_step_point)
            if dist < min_dist:
                min_dist = dist
                save_idx = next_step_idx
                
        already_visited.add(save_idx)
        position_idx = save_idx
        visit_order_indexes.append(position_idx)

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
