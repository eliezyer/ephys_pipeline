import numpy as np
import pandas as pd
from collections import defaultdict

def get_channel_groups(coords, x_threshold=50, y_threshold=50):
    """
    Find groups of coordinate entries that are close together.
    
    An entry belongs to the same group if its closest neighbor is within
    x_threshold in x-coordinate and y_threshold in y-coordinate.
    
    Args:
        npy_path: Path to the input .npy file containing (N, 2) array of coordinates
        x_threshold: Maximum x-coordinate distance for grouping (default: 50)
        y_threshold: Maximum y-coordinate distance for grouping (default: 50)
    
    Returns:
        List of lists, where each inner list contains indices belonging to the same group
    """

    
    print(f"Loaded array with shape: {coords.shape}")
    assert coords.shape[1] == 2, "Expected array with shape (N, 2)"
    
    n_points = coords.shape[0]
    
    # Union-Find data structure for clustering
    parent = list(range(n_points))
    
    def find(x):
        """Find root of x with path compression"""
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        """Union two sets"""
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_x] = root_y
    
    # For each point, find neighbors within threshold and union them
    for i in range(n_points):
        x_i, y_i = coords[i]
        
        # Check all other points
        for j in range(i + 1, n_points):
            x_j, y_j = coords[j]
            
            # Check if within threshold
            x_dist = abs(x_i - x_j)
            y_dist = abs(y_i - y_j)
            
            if x_dist < x_threshold and y_dist < y_threshold:
                union(i, j)
    
    # Group indices by their root
    groups = defaultdict(list)
    for i in range(n_points):
        root = find(i)
        groups[root].append(i)
    
    # Convert to list of lists and sort each group by y-coordinate (most superficial first)
    result = []
    for group in groups.values():
        # Sort indices by y-coordinate in descending order (most superficial first)
        sorted_group = sorted(group, key=lambda idx: coords[idx][1], reverse=True)
        result.append(sorted_group)
    
    print(f"Found {len(result)} groups")
    for idx, group in enumerate(result):
        print(f"  Group {idx}: {len(group)} entries (indices {min(group)}-{max(group)})")
    
    return result


def get_channel_groups_with_regions(coords, region_df, x_threshold=50, y_threshold=50):
    """
    Find groups of coordinate entries that are close together and associate them with brain regions.
    
    Args:
        coords: Array of shape (N, 2) containing x, y coordinates for each channel
        region_df: DataFrame with columns 'region', 'x', and 'y'
                    Example: pd.DataFrame({'region': ['CA1', 'DG', 'CA3'], 
                                          'x': [100, 150, 100], 
                                          'y': [500, 800, 1200]})
        x_threshold: Maximum x-coordinate distance for grouping (default: 50)
        y_threshold: Maximum y-coordinate distance for grouping (default: 50)
    
    Returns:
        Tuple of (channel_groups, region_names):
            - channel_groups: List of lists, where each inner list contains indices belonging to the same group
            - region_names: List of region names where index matches the channel group (None if no match)
    """
    
    # First, get the channel groups using the existing logic
    channel_groups = get_channel_groups(coords, x_threshold, y_threshold)
    
    assert len(channel_groups) == len(region_df), "Number of channel groups must match number of regions"
    # For each channel group, find the range of x and y coordinates
    region_names = []
    
    for group_idx, group in enumerate(channel_groups):
        # Get all coordinates in this group
        group_coords = coords[group]
        
        # Find the range of x and y coordinates
        x_min = np.min(group_coords[:, 0])
        x_max = np.max(group_coords[:, 0])
        y_min = np.min(group_coords[:, 1])
        y_max = np.max(group_coords[:, 1])
        
        print(f"\nGroup {group_idx}:")
        print(f"  X range: [{x_min}, {x_max}]")
        print(f"  Y range: [{y_min}, {y_max}]")
        
        # Find which region(s) fall within this range
        matched_region = None
        for i in range(len(channel_groups)):
            region_x = region_df.iloc[i, 0] # x coordinate
            region_y = region_df.iloc[i, 1] # y coordinate
            if x_min <= region_x <= x_max and y_min <= region_y <= y_max:
                print(f"  Matched region: {region_df.loc[i, 'region']} at ({region_x}, {region_y})")
                matched_region = region_df.loc[i, 'region']
                break  # Take the first match
        region_names.append(matched_region)
    return channel_groups, region_names
