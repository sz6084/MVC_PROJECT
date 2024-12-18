import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage

# Step 1: Simulate 2D slices as a stack (e.g., CT or MRI slices)
def generate_2d_slices(shape=(50, 50), num_slices=20):
    # Create a 3D volume of zeros
    volume = np.zeros((num_slices, *shape))
    
    center = (shape[0] // 2, shape[1] // 2)
    max_radius = 10  # Maximum radius for the circular object
    
    # Define a gradual change in radius across slices
    radii = np.linspace(max_radius, max_radius // 2, num_slices)
    
    for i in range(num_slices):
        radius = radii[i]
        for x in range(shape[0]):
            for y in range(shape[1]):
                # Set a value inside a circular region for each slice
                if (x - center[0]) ** 2 + (y - center[1]) ** 2 < radius**2:
                    volume[i, x, y] = 1
                else:
                    # Set non-circular parts to NaN for transparency
                    volume[i, x, y] = np.nan
    return volume

# Step 2: Stack slices to reconstruct a 3D model
def plot_3d_model(volume):
    # Create a 3D visualization of the volume
    x = np.arange(volume.shape[1])
    y = np.arange(volume.shape[2])
    x, y = np.meshgrid(x, y)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Visualize the 3D volume with non-circular parts transparent
    for i in range(volume.shape[0]):
        z = np.ones_like(x) * i  # Each slice at a different z-level
        
        # Mask NaN values to make them transparent
        slice_data = volume[i]
        mask = ~np.isnan(slice_data)
        facecolors = np.zeros(slice_data.shape + (4,))  # RGBA colors
        facecolors[..., :3] = plt.cm.viridis(slice_data / np.nanmax(slice_data))[:, :, :3]
        facecolors[..., 3] = mask.astype(float) * 0.8  # Transparency for NaN
        
        ax.plot_surface(
            x, y, z, facecolors=facecolors, rstride=1, cstride=1, antialiased=False
        )
    
    ax.set_title('3D Reconstruction of 2D Slices of Gradually Increasing Circles')
    plt.show()

# Step 4: Calculate Surface Area (using gradient of the scalar field)
def calculate_surface_area(volume):
    # Calculate the gradient of the 3D volume
    gradient = np.gradient(np.nan_to_num(volume))
    # Approximate surface area by summing the magnitude of the gradient
    grad_magnitude = np.sqrt(gradient[0]**2 + gradient[1]**2 + gradient[2]**2)
    surface_area = np.sum(grad_magnitude)
    return surface_area

# Main execution
if __name__ == '__main__':
    # Generate 2D slices (e.g., simulate MRI/CT scan slices)
    volume = generate_2d_slices()
    
    # Step 3: Calculate surface area
    surface_area = calculate_surface_area(volume)
    print(f"Calculated Surface Area: {surface_area} units^2")

    # Step 4: Visualize the reconstructed 3D model
    plot_3d_model(volume)
