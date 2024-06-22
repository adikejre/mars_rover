import numpy as np

# Generate Perlin noise
def generate_perlin_noise(shape, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    noise = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            noise[i][j] = noise2d(i / scale, j / scale, octaves, persistence, lacunarity)
    return noise

def noise2d(x, y, octaves, persistence, lacunarity):
    total = 0
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        total += interpolate_noise(x * frequency, y * frequency) * amplitude
        frequency *= lacunarity
        amplitude *= persistence
    return total

def interpolate_noise(x, y):
    x_int = int(x)
    y_int = int(y)
    frac_x = x - x_int
    frac_y = y - y_int
    v1 = smooth_noise(x_int, y_int)
    v2 = smooth_noise(x_int + 1, y_int)
    v3 = smooth_noise(x_int, y_int + 1)
    v4 = smooth_noise(x_int + 1, y_int + 1)
    i1 = interpolate(v1, v2, frac_x)
    i2 = interpolate(v3, v4, frac_x)
    return interpolate(i1, i2, frac_y)

def smooth_noise(x, y):
    corners = (noise(x - 1, y - 1) + noise(x + 1, y - 1) + noise(x - 1, y + 1) + noise(x + 1, y + 1)) / 16
    sides = (noise(x - 1, y) + noise(x + 1, y) + noise(x, y - 1) + noise(x, y + 1)) / 8
    center = noise(x, y) / 4
    return corners + sides + center

def interpolate(a, b, x):
    ft = x * np.pi
    f = (1 - np.cos(ft)) * 0.5
    return a * (1 - f) + b * f

def noise(x, y):
    n = x + y * 57
    n = (n << 13) ^ n
    return (1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0)

# Generate Perlin noise
shape = (100, 100)  # Adjust grid size as needed
scale = 100.0  # Adjust scale to control the "zoom" level of the noise
octaves = 6  # Adjust octaves to control the level of detail
persistence = 0.5  # Adjust persistence to control the roughness
lacunarity = 2.0  # Adjust lacunarity to control the frequency
terrain = generate_perlin_noise(shape, scale, octaves, persistence, lacunarity)

# Normalize terrain values to range [0, 1]
terrain = (terrain - np.min(terrain)) / (np.max(terrain) - np.min(terrain))

# Visualize the terrain
import matplotlib.pyplot as plt
plt.imshow(terrain, cmap='terrain', interpolation='bilinear')
plt.colorbar()
plt.show()
