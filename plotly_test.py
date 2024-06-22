import numpy as np
import rasterio
import plotly.graph_objects as go

# Path to the DEM file
# dem_path = 'D:\\adity\\Downloads\\b01_009894_1665_xi_13s042w_b02_010606_1666_xn_13s042w_tied-dem.tif'
dem_path = 'D:\\adity\\Downloads\\b18_016833_1714_xi_08s124w_g02_019035_1714_xn_08s124w_tied-dem.tif'




# # Read the DEM file
# with rasterio.open(dem_path) as src:
#     dem_data = src.read(1)  # Read the first band
#     transform = src.transform

# # Get the dimensions of the DEM data
# height, width = dem_data.shape

# # Generate x, y coordinates based on the transform
# x_coords, y_coords = np.meshgrid(
#     np.arange(width),
#     np.arange(height)
# )

# # Apply the affine transform to x and y coordinates
# x_coords = x_coords * transform[0] + transform[2]
# y_coords = y_coords * transform[4] + transform[5]

# # Normalize the data so that the first value is (0,0,0)
# x0, y0, z0 = x_coords[0, 0], y_coords[0, 0], dem_data[0, 0]
# x_coords -= x0
# y_coords -= y0
# dem_data -= z0

# # Scale x and y coordinates by dividing by 18
# x_coords /= 18
# y_coords /= 18

# # Verify the processed data
# print("x_coords shape:", x_coords.shape)
# print("y_coords shape:", y_coords.shape)
# print("dem_data shape:", dem_data.shape)

# # Ensure there are no NaN or Inf values
# x_coords = np.nan_to_num(x_coords)
# y_coords = np.nan_to_num(y_coords)
# dem_data = np.nan_to_num(dem_data)

# # Create a surface plot
# fig = go.Figure(data=[go.Surface(z=dem_data, x=x_coords, y=y_coords, colorscale='Viridis')])
# fig.update_layout(title='3D Terrain Map', autosize=True,
#                   scene=dict(xaxis_title='Scaled X Coordinate (m)',
#                              yaxis_title='Scaled Y Coordinate (m)',
#                              zaxis_title='Normalized Elevation (m)'))
# fig.show()



# with rasterio.open(dem_path) as dataset:
#     # Read the DEM data (elevation values)
#     dem_data = dataset.read(1)  # Assuming the DEM data is in the first band
#     # Get the affine transformation
#     transform = dataset.transform

# # Create grid indices
# height, width = dem_data.shape
# x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height))

# # Apply the affine transformation to the grid indices
# x_coords = transform[0] * x_indices + transform[1] * y_indices + transform[2]
# y_coords = transform[3] * x_indices + transform[4] * y_indices + transform[5]

# # Normalize the coordinates
# x_coords -= x_coords.min()
# y_coords -= y_coords.min()


# import plotly.graph_objects as go

# # Create a surface plot using Plotly
# fig = go.Figure(data=[go.Surface(z=dem_data, x=x_coords, y=y_coords, colorscale='Viridis')])
# fig.update_layout(title='3D Terrain Map', autosize=True,
#                   scene=dict(xaxis_title='Scaled X Coordinate',
#                              yaxis_title='Scaled Y Coordinate',
#                              zaxis_title='Normalized Elevation'))
# fig.show()


# Sample data
# x = np.linspace(-5, 5, 100)
# y = np.linspace(-5, 5, 100)
# x, y = np.meshgrid(x, y)
# z = np.sin(np.sqrt(x**2 + y**2))

# # Display sample data
# print("Sample x data:")
# print(x[:5, :5])
# print("Sample y data:")
# print(y[:5, :5])
# print("Sample z data:")
# print(z[:5, :5])

# # Create a plotly surface plot
# fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis')])
# fig.update_layout(title='3D Terrain Map', autosize=True,
#                   scene=dict(xaxis_title='X Coordinate',
#                              yaxis_title='Y Coordinate',
#                              zaxis_title='Elevation'))
# fig.show()















# Open the DEM file
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

# # Print the first few values to verify normalization and scaling
# print("Normalized and scaled x_coords:\n", x_coords[:5, :5])
# print("Normalized and scaled y_coords:\n", y_coords[:5, :5])


print("Sample x_coords:\n", x_coords)
print("Sample y_coords:\n", y_coords)

# print(np.shape(x_coords))
# print(np.shape(y_coords))

# lower_bound = 16060
# upper_bound = 16110

# # Create a boolean mask
# mask = (x_coords >= lower_bound) & (x_coords <= upper_bound)
# indices = np.where(mask)

# # Combine the indices into a list of tuples
# xindices_list = list(zip(indices[0], indices[1]))

# ylower_bound = 12410
# yupper_bound = 12430

# # Create a boolean mask
# mask = (y_coords >= ylower_bound) & (y_coords <= yupper_bound)
# indices = np.where(mask)

# # Combine the indices into a list of tuples
# yindices_list = list(zip(indices[0], indices[1]))

# print("list")
# print(x_coords[800])
# # print(yindices_list)

# (1445, 1905)

# row_start, row_end = 395, 745
# col_start, col_end = 700, 1050

# row_start, row_end = 700, 900
# col_start, col_end = 700, 900

row_start, row_end = 600, 900
col_start, col_end = 700, 900


# # Slice the data
dem_data_subset = dem_data[row_start:row_end, col_start:col_end]
x_coords_subset = x_coords[row_start:row_end, col_start:col_end]
y_coords_subset = y_coords[row_start:row_end, col_start:col_end]

print("hii")
print(dem_data_subset.max())
print(dem_data_subset.min())

dem_data_subset_normalized = (dem_data_subset - dem_data_subset.min()) / (dem_data_subset.max() - dem_data_subset.min())

# Create a surface plot using Plotly
fig = go.Figure(data=[go.Surface(z=dem_data_subset_normalized, x=x_coords_subset, y=y_coords_subset, colorscale='Viridis')])
fig.update_layout(title='3D Terrain Map', autosize=True,
                  scene=dict(xaxis_title='Scaled X Coordinate',
                             yaxis_title='Scaled Y Coordinate',
                             zaxis_title='Elevation'))
fig.show()




