import numpy as np
import rasterio
import plotly.graph_objects as go

# Path to the DEM file
# dem_path = 'D:\\adity\\Downloads\\b01_009894_1665_xi_13s042w_b02_010606_1666_xn_13s042w_tied-dem.tif'
# https://asc-pds-individual-investigations.s3.us-west-2.amazonaws.com/mars_mro_hirise_explorationzones_day_2023/index.html
dem_path = 'b18_016833_1714_xi_08s124w_g02_019035_1714_xn_08s124w_tied-dem.tif'



def replace_outliers_with_adjacent_average(arr, threshold):
    non_zero_values = arr[arr != 0]
    
    # Calculate mean and standard deviation of non-zero values
    mean_non_zero = np.mean(non_zero_values)
    std_non_zero = np.std(non_zero_values)
    
    # Identify values within one standard deviation of the mean
    within_std_values = non_zero_values[np.abs(non_zero_values - mean_non_zero) <= std_non_zero]
    
    # Calculate the mean of these values
    mean_within_std = np.mean(within_std_values)

    outliers = np.abs(arr - mean_within_std) > threshold * np.std(arr)  # Identify outliers using z-score
    print("OUTLIERS")
    print(outliers)

    
    
    # Iterate over each outlier
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if outliers[i, j]:
                adjacent_values = []
                # Collect adjacent values (horizontally and vertically)
                for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    if 0 <= i + di < arr.shape[0] and 0 <= j + dj < arr.shape[1]:
                        adjacent_values.append(arr[i + di, j + dj])
                # Replace outlier with the average of adjacent values
                # arr[i, j] = np.mean(arr[arr != 0])
                arr[i, j] = mean_within_std
    
    return arr

# Specify threshold for z-score (e.g., 2 for removing values > 2 standard deviations from mean)
threshold = 2

with rasterio.open(dem_path) as dataset:
    dem_data = dataset.read(1)  # Assuming the DEM data is in the first band
    transform = dataset.transform

# Print the shape of the DEM data
print("DEM data shape:", dem_data.shape)



height, width = dem_data.shape
x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height))
print(x_indices)
print(np.shape(x_indices))
# print(y_indices)

# # Apply the affine transformation to the grid indices
x_coords = transform[0] * x_indices + transform[1] * y_indices + transform[2]
y_coords = transform[3] * x_indices + transform[4] * y_indices + transform[5]


# # Print the first few values to verify
# print("Sample x_coords:\n", x_coords[:5, :5])
# print("Sample y_coords:\n", y_coords[:5, :5])


# # Normalize the coordinates
x_coords -= x_coords.min()
y_coords -= y_coords.min()

# # Scale the coordinates (if needed)
x_coords /= 18
y_coords /= 18




# (1445, 1905)

# row_start, row_end = 395, 745
# col_start, col_end = 700, 1050

# row_start, row_end = 700, 900
# col_start, col_end = 700, 900

# row_start, row_end = 600, 900
# col_start, col_end = 600, 900

row_start, row_end = 1200, 1355
col_start, col_end = 250, 600



# # Slice the data
dem_data_subset = dem_data[row_start:row_end, col_start:col_end]
x_coords_subset = x_coords[row_start:row_end, col_start:col_end]
y_coords_subset = y_coords[row_start:row_end, col_start:col_end]

dem_data_subset_normalized = (dem_data_subset - dem_data_subset.min()) / (dem_data_subset.max() - dem_data_subset.min())


# Replace outliers with the average of adjacent values
dem_data_subset_cleaned= replace_outliers_with_adjacent_average(dem_data_subset_normalized, threshold)

print("hii")
print(dem_data_subset_cleaned.max())
print(dem_data_subset_cleaned.min())


# Create a surface plot using Plotly
fig = go.Figure(data=[go.Surface(z=dem_data_subset_normalized, x=x_coords_subset, y=y_coords_subset, colorscale='Viridis')])
fig.update_layout(title='3D Terrain Map', autosize=True,
                  scene=dict(xaxis_title='Scaled X Coordinate',
                             yaxis_title='Scaled Y Coordinate',
                             zaxis_title='Elevation'))
fig.show()





