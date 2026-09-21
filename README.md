# Map, Data Association & Path Planning

## The problem

Imagine we are building a robot car that moves through a flat environment. Its internal map contains the 2D positions of landmarks, for example, cirles printed on the floor. While driving, the car can observe some landmarks again. These newly observed landmarks are noisy, arrive in a shuffled order, and they are not granted for each landmark we have in the original map.

The car uses these new measurements to **update its internal map**: after determining which observation belongs to which landmark, it adjusts the stored position of that landmark toward the new measurement. It must then plan a route through the updated landmarks.

Complete the three functions in `solution.py`, in this order:

1. **`associate(map_points, observations)`**: determine the correspondence between the car's known landmarks and its new sensor measurements. Return `(map_index, observation_index)` pairs, where each pair identifies one landmark and the observation believed to come from it. Note that an observation can't be used more than once.
2. **`correct_map(map_points, observations, associations)`**: create the car's updated internal map using the associations. For each matched landmark, set its new position to the midpoint of its previous map position and its observed position: this combines the prior estimate with the new, noisy measurement. A landmark that was not observed keeps its previous position. Return a map with the same length and landmark order as the input.
3. **`find_path(corrected_map)`**: plan an ordered visit to all landmarks in the updated map. Return the landmark indices in the order the car should visit them, ensuring that every node appears exactly once.

**Important notes**: Keep the required function names and signatures unchanged. Your solution may be evaluated with parameters and random seeds different from the defaults.

The expected outcome is a complete, correct implementation of the three required steps **without using AI**. This is a capability test: candidates may be asked to justify any design or implementation choice.

## Submission

Fork this repository into a **public repository** and complete the assignment there. When you are finished, share the link to your public repository as your submission.

## Interesting things to explore

Once the core pipeline works, you may explore more advanced approaches and compare their behavior:

- **Path quality:** try to reduce the route's total length, calculated as the sum of the Euclidean distances between consecutive landmarks. Use `path_length(...)` to compare different orderings and heuristics.
- **Path geometry:** investigate how to construct a route whose segments do not cross, except where consecutive segments meet at their shared landmark. Check the result with `visualize_path(...)`.

## Generation and visualization tools

Install the dependency and run the pipeline from the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 solution.py
```

`solution.py` already uses the provided tools:

- `generate_map(...)` creates the landmark map.
- `generate_observations(...)` creates noisy, incomplete, shuffled observations.
- `visualize_map_n_observations(...)` shows the generated map and observations.
- `visualize_association_n_correction(...)` shows your associations and corrected map.
- `visualize_path(...)` shows the path returned by `find_path`.

The script displays the first visualization, then continues through the later steps as you implement them. Adjust the constants at the top of `solution.py` to experiment with different map sizes, noise levels, observation rates, and seeds. Use `path_length(...)` in `solution.py` to measure a path when working on Bonus 1.
