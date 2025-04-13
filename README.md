# SafePath: Safety-Aware Routing in Chicago

This project calculates pedestrian routes in Chicago that balance both **distance** and **safety**, using real-world crime, lighting, and emergency infrastructure data.

## How to Run

1. Open the `safe_path_test.py` script.
2. Set your desired start and end coordinates on the following lines:

```python
start_coords = (41.7796, -87.6636)  # West Garfield Park  
end_coords = (41.7681, -87.6435)    # Austin
```

## Results

1. Output will be stored in `multi_lambda_paths_map.html`
2. Type `open multi_lambda_paths_map.html` in your terminal to open it
