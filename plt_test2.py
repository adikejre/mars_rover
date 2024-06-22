import numpy as np
import rasterio
import plotly.graph_objects as go

# Path to the DEM file
# dem_path = 'D:\\adity\\Downloads\\b01_009894_1665_xi_13s042w_b02_010606_1666_xn_13s042w_tied-dem.tif'
dem_path = 'D:\\adity\\Downloads\\b18_016833_1714_xi_08s124w_g02_019035_1714_xn_08s124w_tied-dem.tif'


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


x_start_val, x_end_val = 700, 1050
y_start_val, y_end_val = 1050, 700



col_start = np.searchsorted(x_coords[0, :], x_start_val, side='left')
col_end = np.searchsorted(x_coords[0, :], x_end_val, side='right')

# Find the corresponding row indices for y_coords
row_start = np.searchsorted(y_coords[:, 0], y_end_val, side='right')
row_end = np.searchsorted(y_coords[:, 0], y_start_val, side='left')

# Correct the row order for y_coords since y decreases
row_start, row_end = row_start, row_end

# # Find the corresponding column indices for x_coords
# col_start = np.searchsorted(x_coords[0, :], x_start_val)
# col_end = np.searchsorted(x_coords[0, :], x_end_val)

# # Find the corresponding row indices for y_coords
# row_start = np.searchsorted(y_coords[:, 0], y_start_val, side='right') - 1
# row_end = np.searchsorted(y_coords[:, 0], y_end_val, side='right') - 1

# # Correct the slice range for y_coords since y decreases
# row_start, row_end = row_end, row_start




# Now slice the arrays
dem_data_subset = dem_data[row_start:row_end, col_start:col_end]
x_coords_subset = x_coords[row_start:row_end, col_start:col_end]
y_coords_subset = y_coords[row_start:row_end, col_start:col_end]

# Print shapes to verify
print("Subset DEM Data shape:", dem_data_subset.shape)
print("Subset X Coords shape:", x_coords_subset.shape)
print("Subset Y Coords shape:", y_coords_subset.shape)

print(row_start)
print(row_end)
print(col_start)
print(col_end)



# Flatten the subset data for scatter plot
x_flat = x_coords_subset.flatten()
y_flat = y_coords_subset.flatten()
z_flat = dem_data_subset.flatten()

# Create a 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=x_flat, y=y_flat, z=z_flat,
    mode='markers',
    marker=dict(
        size=2,
        color=z_flat,                # set color to the z values
        colorscale='cividis',        # choose a colorscale
        opacity=0.8
    )
)])

# Update layout for better visualization
fig.update_layout(
    scene=dict(
        xaxis_title='X Coordinate',
        yaxis_title='Y Coordinate',
        zaxis_title='Elevation (Z)',
    ),
    title='3D Scatter Plot of DEM Subset'
)

# Show plot
fig.show()