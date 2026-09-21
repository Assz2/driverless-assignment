"""Visualization utilities for maps, observations, associations and paths.

These helpers are provided as part of the assignment code: you should not
need to modify them, but you are encouraged to read them, since they are
the executable documentation of the data formats expected at every step.

Every function draws a matplotlib figure with an *equal* aspect ratio (so
that distances and crossings look geometrically faithful), optionally saves
it to ``save_path``, and opens the figure window unless ``show=False``.
"""

import warnings
from numbers import Integral
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from map_generation import Map, Point

# Visual style: the colors are those required by the assignment statement.
COLOR_MAP = "red"              # map points
COLOR_OBSERVATION = "orange"   # observations
COLOR_CORRECTED = "purple"     # corrected map points
COLOR_ASSOCIATION = "dimgray"  # dotted association lines
COLOR_PATH = "green"           # path arrows
COLOR_BOUNDARY = "lightgray"   # map area boundary


def _setup_axes(
    ax: plt.Axes,
    x_range: float,
    y_range: float,
    draw_boundary: bool = True,
) -> None:
    """Configure axis limits, aspect ratio, grid and optional area boundary."""
    half_x = x_range / 2.0
    half_y = y_range / 2.0
    if draw_boundary:
        ax.add_patch(
            Rectangle(
                xy=(-half_x, -half_y),
                width=x_range,
                height=y_range,
                fill=False,
                edgecolor=COLOR_BOUNDARY,
                linestyle="--",
                linewidth=1.0,
                zorder=1,
            )
        )
    margin_x = 0.06 * x_range
    margin_y = 0.06 * y_range
    ax.set_xlim(-half_x - margin_x, half_x + margin_x)
    ax.set_ylim(-half_y - margin_y, half_y + margin_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)


def _finish_figure(
    fig: plt.Figure, save_path: Optional[str], show: bool
) -> None:
    """Save the figure if requested, then show it (or close it)."""
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _scatter_points(
    ax: plt.Axes,
    points: List[Point],
    color: str,
    marker: str,
    label: str,
    size: float = 45,
    zorder: float = 3,
) -> None:
    """Scatter a list of ``(x, y)`` points, if any."""
    if points:
        xs, ys = zip(*points)
        ax.scatter(xs, ys, c=color, marker=marker, s=size, label=label, zorder=zorder)


def _add_legend(ax: plt.Axes, extra_handles=(), extra_labels=()) -> None:
    """Add a legend from the labeled artists, plus optional proxy handles."""
    handles, labels = ax.get_legend_handles_labels()
    handles = handles + list(extra_handles)
    labels = labels + list(extra_labels)
    if handles:
        ax.legend(handles, labels, loc="upper right")


def _plot_range(map_points: Map) -> Tuple[float, float]:
    """Return the (x_range, y_range) of a box centered on 0 covering all points."""
    if not map_points:
        raise ValueError("Cannot visualize a path over an empty map.")
    x_range = 2.0 * max(abs(x) for x, _ in map_points)
    y_range = 2.0 * max(abs(y) for _, y in map_points)
    return max(x_range, 1.0), max(y_range, 1.0)


def _check_associations(
    associations: List[Tuple[int, int]],
    num_map_points: int,
    num_observations: int,
) -> None:
    """Validate association couples: raise on bad indices, warn on duplicates."""
    seen_map = set()
    seen_obs = set()
    for couple in associations:
        try:
            map_idx, obs_idx = couple
        except (TypeError, ValueError):
            raise ValueError(
                f"Association {couple!r} is not a couple "
                "(map_index, observation_index)."
            ) from None
        if not isinstance(map_idx, Integral) or not isinstance(obs_idx, Integral):
            raise ValueError(f"Association {couple!r}: both indices must be integers.")
        if not 0 <= map_idx < num_map_points:
            raise ValueError(
                f"Association {couple!r}: map index {map_idx} is out of range "
                f"(the map has {num_map_points} points)."
            )
        if not 0 <= obs_idx < num_observations:
            raise ValueError(
                f"Association {couple!r}: observation index {obs_idx} is out of "
                f"range (there are {num_observations} observations)."
            )
        if map_idx in seen_map:
            warnings.warn(
                f"Map point {map_idx} is associated more than once: its "
                "corrected position is not well defined."
            )
        if obs_idx in seen_obs:
            warnings.warn(
                f"Observation {obs_idx} is associated more than once."
            )
        seen_map.add(map_idx)
        seen_obs.add(obs_idx)


def visualize_map_n_observations(
    map_points: Map,
    observations: List[Point],
    x_range: float,
    y_range: float,
    title: str = "Map and observations",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Visualize a map together with its observations.

    The map points are drawn as red circles and the observations as orange
    squares, inside the rectangular map area (dashed boundary) defined by
    ``x_range`` and ``y_range``.

    Args:
        map_points: The map points (red circles).
        observations: The observations (orange squares).
        x_range: Total width of the map area along x, centered on 0.
        y_range: Total height of the map area along y, centered on 0.
        title: Title of the figure.
        save_path: If given, the figure is saved to this path (e.g.
            ``"step2_map_observations.png"``).
        show: If False, the figure window is not opened (useful when you
            only want to save the figure).
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    _setup_axes(ax, x_range, y_range)
    _scatter_points(ax, map_points, COLOR_MAP, "o", "Map points")
    _scatter_points(ax, observations, COLOR_OBSERVATION, "s", "Observations", size=40)
    _add_legend(ax)
    ax.set_title(title)
    _finish_figure(fig, save_path, show)


def visualize_association_n_correction(
    map_points: Map,
    observations: List[Point],
    corrected_map: Map,
    associations: List[Tuple[int, int]],
    x_range: float,
    y_range: float,
    title: str = "Data association and map correction",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Visualize the data-association couples and the corrected map.

    Draws, in order:

    * the map points as red circles;
    * the observations as orange squares;
    * one gray dotted line per association couple ``(map_index,
      observation_index)``, connecting the paired map point and observation;
    * the corrected map points as purple diamonds.

    Args:
        map_points: The original map points (red circles).
        observations: The observations (orange squares).
        corrected_map: The corrected map (purple diamonds); expected to have
            the same length and order as ``map_points``.
        associations: The association couples ``(map_index, observation_index)``.
        x_range: Total width of the map area along x, centered on 0.
        y_range: Total height of the map area along y, centered on 0.
        title: Title of the figure.
        save_path: If given, the figure is saved to this path.
        show: If False, the figure window is not opened.

    Raises:
        ValueError: If an association couple is malformed or refers to a
            map/observation index that does not exist.
    """
    _check_associations(associations, len(map_points), len(observations))
    if len(corrected_map) != len(map_points):
        warnings.warn(
            f"corrected_map has {len(corrected_map)} points but map_points has "
            f"{len(map_points)}: the corrected map is expected to keep the same "
            "length and order as the map."
        )

    fig, ax = plt.subplots(figsize=(8, 8))
    _setup_axes(ax, x_range, y_range)
    _scatter_points(ax, map_points, COLOR_MAP, "o", "Map points")
    _scatter_points(ax, observations, COLOR_OBSERVATION, "s", "Observations", size=40)

    for map_idx, obs_idx in associations:
        map_x, map_y = map_points[map_idx]
        obs_x, obs_y = observations[obs_idx]
        ax.plot(
            [map_x, obs_x],
            [map_y, obs_y],
            linestyle=":",
            color=COLOR_ASSOCIATION,
            linewidth=1.2,
            zorder=2,
        )

    _scatter_points(
        ax, corrected_map, COLOR_CORRECTED, "D", "Corrected map", size=50, zorder=4
    )

    _add_legend(
        ax,
        extra_handles=[
            Line2D([0], [0], linestyle=":", color=COLOR_ASSOCIATION, linewidth=1.2)
        ],
        extra_labels=["Associations"],
    )
    ax.set_title(title)
    _finish_figure(fig, save_path, show)


def visualize_path(
    map_points: Map,
    path: List[int],
    title: str = "Path over the map nodes",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Visualize a path over the nodes of a map.

    The map points are drawn in gray; consecutive nodes of ``path`` are
    connected by green arrows, from node ``path[i]`` to node ``path[i + 1]``.
    The plot range is computed from the points themselves, so the function
    works with the original map as well as with the corrected map.

    Args:
        map_points: The map points (gray circles): original or corrected map.
        path: The node indices (into ``map_points``) in visit order.
        title: Title of the figure.
        save_path: If given, the figure is saved to this path.
        show: If False, the figure window is not opened.

    Raises:
        ValueError: If ``path`` refers to a node index that does not exist.
    """
    for node in path:
        if not isinstance(node, Integral) or not 0 <= node < len(map_points):
            raise ValueError(
                f"Path node {node!r} is not a valid index into the map "
                f"(which has {len(map_points)} points)."
            )
    duplicates = {node for node in path if path.count(node) > 1}
    if duplicates:
        warnings.warn(
            f"The path visits these nodes more than once: {sorted(duplicates)}. "
            "A valid path must include each node exactly once."
        )

    fig, ax = plt.subplots(figsize=(8, 8))
    x_range, y_range = _plot_range(map_points)
    _setup_axes(ax, x_range, y_range, draw_boundary=False)
    _scatter_points(ax, map_points, "gray", "o", "Map nodes")

    for start, end in zip(path, path[1:]):
        ax.annotate(
            "",
            xy=map_points[end],
            xytext=map_points[start],
            arrowprops=dict(
                arrowstyle="-|>",
                color=COLOR_PATH,
                linewidth=1.8,
                shrinkA=5,
                shrinkB=5,
                mutation_scale=16,
            ),
            zorder=2,
        )

    _add_legend(
        ax,
        extra_handles=[Line2D([0], [0], color=COLOR_PATH, linewidth=1.8)],
        extra_labels=["Path"],
    )
    ax.set_title(title)
    _finish_figure(fig, save_path, show)
