import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage

# Step 1: Simulate 2D slices as a stack (e.g., CT or MRI slices)
def generate_2d_slices(shape=(50, 50), num_slices=20):
    # Create a 3D volume of random data with some shape
    volume = np.random.rand(num_slices, *shape)
    
    # For simplicity, we're simulating a circular object in the middle of the volume
    center = (num_slices // 2, shape[0] // 2, shape[1] // 2)
    radius = 10
    
    for i in range(num_slices):
        for x in range(shape[0]):
            for y in range(shape[1]):
                # Set a value inside a circular region of each slice
                if (x - shape[0] // 2) ** 2 + (y - shape[1] // 2) ** 2 < radius**2:
                    volume[i, x, y] = 1
    return volume

# Step 2: Stack slices to reconstruct a 3D model
def plot_3d_model(volume):
    # Create a 3D visualization of the volume
    x = np.arange(volume.shape[1])
    y = np.arange(volume.shape[2])
    x, y = np.meshgrid(x, y)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Visualize the 3D volume
    for i in range(volume.shape[0]):
        z = np.ones_like(x) * i  # Each slice at a different z-level
        ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(volume[i]), rstride=1, cstride=1, alpha=0.5)
    
    ax.set_title('3D Reconstruction of 2D Slices')
    plt.show()

# Step 3: Calculate Volume (using integration on 3D grid)
def calculate_volume(volume, dx=1, dy=1, dz=1):
    return np.sum(volume) * dx * dy * dz

# Step 4: Calculate Surface Area (using gradient of the scalar field)
def calculate_surface_area(volume):
    # Calculate the gradient of the 3D volume
    gradient = np.gradient(volume)
    # Approximate surface area by summing the magnitude of the gradient
    grad_magnitude = np.sqrt(gradient[0]**2 + gradient[1]**2 + gradient[2]**2)
    surface_area = np.sum(grad_magnitude)
    return surface_area

# Main execution
if __name__ == '__main__':
    # Generate 2D slices (e.g., simulate MRI/CT scan slices)
    volume = generate_2d_slices()
    
    # Step 2: Calculate volume
    volume_value = calculate_volume(volume)
    print(f"Calculated Volume: {volume_value} cubic units")
    
    # Step 3: Calculate surface area
    surface_area = calculate_surface_area(volume)
    print(f"Calculated Surface Area: {surface_area} units^2")

    # Step 4: Visualize the reconstructed 3D model
    plot_3d_model(volume)