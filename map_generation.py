"""Generation of random landmark maps and of the corresponding observations.

This module is part of the provided assignment code: you should not need to
modify it. It exposes the two data-generation entry points used throughout
the assignment:

* :func:`generate_map` -- a random map of ``(x, y)`` points;
* :func:`generate_observations` -- noisy, incomplete, *shuffled* observations
  of a map.

A *map* is represented as a plain ``list`` of ``(x, y)`` tuples: no custom
class, no hidden state. The position of a point in the list is its node
index, and it is what association couples and paths refer to.
"""

import math
import random
from typing import List, Optional, Tuple

#: A single 2D point, as an ``(x, y)`` tuple of floats.
Point = Tuple[float, float]

#: A map: an ordered list of 2D points (the position in the list is the node index).
Map = List[Point]

# Safety valve for ``generate_map``: after this many consecutive rejections
# the minimum-distance constraint is relaxed, so that generation always
# terminates even for unrealistically dense maps.
_MAX_RESAMPLE_ATTEMPTS = 1000


def generate_map(
    num_points: int,
    x_range: float,
    y_range: float,
    min_point_distance: float = 1.0,
    seed: Optional[int] = None,
) -> Map:
    """Generate a map of ``num_points`` random points.

    The points are sampled uniformly at random inside the rectangle centered
    on the origin with total width ``x_range`` and total height ``y_range``::

        x in [-x_range / 2, +x_range / 2]
        y in [-y_range / 2, +y_range / 2]

    Args:
        num_points: Number of points of the map.
        x_range: Total width of the map area along x (centered on 0).
        y_range: Total height of the map area along y (centered on 0).
        min_point_distance: Minimum distance enforced between any two map
            points, so that the data-association problem stays well posed.
            Candidates that violate it are re-sampled; if the requested map
            is too dense for this constraint, it is relaxed after enough
            attempts so that generation always terminates.
        seed: Optional seed for a reproducible map.

    Returns:
        The map: a list of ``num_points`` ``(x, y)`` tuples.
    """
    if num_points < 1:
        raise ValueError(f"num_points must be at least 1, got {num_points}")
    if x_range <= 0.0 or y_range <= 0.0:
        raise ValueError(
            f"x_range and y_range must be positive, got {x_range} and {y_range}"
        )
    if min_point_distance < 0.0:
        raise ValueError(
            f"min_point_distance must be non-negative, got {min_point_distance}"
        )

    rng = random.Random(seed)
    half_x = x_range / 2.0
    half_y = y_range / 2.0

    points: Map = []
    consecutive_rejections = 0
    while len(points) < num_points:
        candidate = (rng.uniform(-half_x, half_x), rng.uniform(-half_y, half_y))
        far_enough = all(
            math.dist(candidate, point) >= min_point_distance for point in points
        )
        if far_enough or consecutive_rejections >= _MAX_RESAMPLE_ATTEMPTS:
            points.append(candidate)
            consecutive_rejections = 0
        else:
            consecutive_rejections += 1
    return points


def generate_observations(
    map_points: Map,
    noise_std: float = 0.25,
    observation_rate: float = 0.8,
    seed: Optional[int] = None,
) -> List[Point]:
    """Generate the observations of a given map.

    Each map point independently produces an observation with probability
    ``observation_rate``. Every produced observation is the corresponding
    map point perturbed with independent Gaussian noise (mean 0, standard
    deviation ``noise_std``) on each axis.

    The returned list is **shuffled**: the position of an observation in the
    list carries *no* information about which map point it was generated
    from. Recovering that correspondence is exactly the data-association
    task of the assignment (Step 3).

    Args:
        map_points: The map to observe.
        noise_std: Standard deviation of the per-axis Gaussian noise.
        observation_rate: Probability that a given map point produces an
            observation (must be in ``[0, 1]``).
        seed: Optional seed for reproducible observations.

    Returns:
        The observations: a list of ``(x, y)`` tuples with length between
        ``0`` and ``len(map_points)``.
    """
    if not 0.0 <= observation_rate <= 1.0:
        raise ValueError(
            f"observation_rate must be in [0, 1], got {observation_rate}"
        )
    if noise_std < 0.0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")

    rng = random.Random(seed)
    observations: List[Point] = []
    for x, y in map_points:
        if rng.random() < observation_rate:
            observations.append(
                (
                    x + rng.gauss(0.0, noise_std),
                    y + rng.gauss(0.0, noise_std),
                )
            )
    rng.shuffle(observations)
    return observations
