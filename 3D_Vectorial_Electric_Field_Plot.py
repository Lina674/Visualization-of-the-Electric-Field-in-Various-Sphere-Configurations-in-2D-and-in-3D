import numpy as np
import plotly.graph_objects as go

# Constants
k = 8.99e9  # Coulomb's constant
q = float(input('Enter the desired charge: '))   # Charge of the spherical distribution (in Coulombs)
r = float(input('Enter the desired radius: '))    # Radius of the spherical distribution (in meters)

# Define the grid for the 3D plot
x = np.linspace(-(2*r), (2*r), 20)
y = np.linspace(-(2*r), (2*r), 20)
z = np.linspace(-(2*r), (2*r), 20)
X, Y, Z = np.meshgrid(x, y, z)

# Calculate the electric field at each point in the grid
R = np.sqrt(X**2 + Y**2 + Z**2)
Ex = np.zeros_like(X)
Ey = np.zeros_like(Y)
Ez = np.zeros_like(Z)

# Create a meshgrid for drawing the sphere outlines
phi = np.linspace(0, 2 * np.pi, 100)
theta = np.linspace(0, np.pi, 100)
phi, theta = np.meshgrid(phi, theta)

# Calculate spherical coordinates for the outlines of the spheres
inner_sphere_x = r * np.sin(theta) * np.cos(phi)
inner_sphere_y = r * np.sin(theta) * np.sin(phi)
inner_sphere_z = r * np.cos(theta)

# Create a 3D scatter plot for the spheres producing the electric field
inner_sphere_outline = go.Scatter3d(
    x=inner_sphere_x.flatten(),
    y=inner_sphere_y.flatten(),
    z=inner_sphere_z.flatten(),
    mode='lines',
    line=dict(width=1, color='white'),
    name=''
)


# Calculate electric field outside the central sphere
outside_sphere = (R >= r)
Ex[outside_sphere] = k * q * X[outside_sphere] / (R[outside_sphere]**3 + 1e-10)  # Added a small value to avoid division by zero
Ey[outside_sphere] = k * q * Y[outside_sphere] / (R[outside_sphere]**3 + 1e-10)
Ez[outside_sphere] = k * q * Z[outside_sphere] / (R[outside_sphere]**3 + 1e-10)

# Create an interactive 3D plot
fig = go.Figure(data=go.Cone(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), u=Ex.flatten(), v=Ey.flatten(), w=Ez.flatten(), sizemode="absolute", colorbar=dict(title="Electric Field")))
fig.update_layout(scene=dict(aspectmode="data"))
fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=1)))

# Set axis labels and title
fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
fig.update_layout(title='Interactive Electric Field Vectors Outside a Uniform Spherical Charge Distribution')

# Add the central charge outlines
fig.add_trace(inner_sphere_outline)

# Show the interactive plot
fig.show()
