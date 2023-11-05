import numpy as np
import plotly.graph_objects as go

# Constants
k = 8.99e9  # Coulomb's constant

# Radii of the spheres
R_inner = float(input('Enter the radius of the inner shell: ')) # Inner sphere radius
R_outer = float(input('Enter the radius of the outer shell: '))  # Outer sphere radius

# Charge values for the inner and outer spheres
q_inner = float(input('Enter the charge of the inner shell: '))  # Charge on the inner sphere (in Coulombs)
q_outer = float(input('Enter the charge of the outer shell: '))  # Charge on the outer sphere (in Coulombs)


# Define the grid for the 3D plot
x = np.linspace(-R_outer, R_outer, 20)
y = np.linspace(-R_outer, R_outer, 20)
z = np.linspace(-R_outer, R_outer, 20)
X, Y, Z = np.meshgrid(x, y, z)

# Calculate the electric field between the spheres
R = np.sqrt(X**2 + Y**2 + Z**2)
Ex = np.zeros_like(X)
Ey = np.zeros_like(Y)
Ez = np.zeros_like(Z)

# Create a meshgrid for drawing the sphere outlines
phi = np.linspace(0, 2 * np.pi, 100)
theta = np.linspace(0, np.pi, 100)
phi, theta = np.meshgrid(phi, theta)

# Calculate spherical coordinates for the outlines of the spheres
inner_sphere_x = R_inner * np.sin(theta) * np.cos(phi)
inner_sphere_y = R_inner * np.sin(theta) * np.sin(phi)
inner_sphere_z = R_inner * np.cos(theta)

outer_sphere_x = R_outer * np.sin(theta) * np.cos(phi)
outer_sphere_y = R_outer * np.sin(theta) * np.sin(phi)
outer_sphere_z = R_outer * np.cos(theta)

# Create a 3D scatter plot for the spheres producing the electric field
inner_sphere_outline = go.Scatter3d(
    x=inner_sphere_x.flatten(),
    y=inner_sphere_y.flatten(),
    z=inner_sphere_z.flatten(),
    mode='lines',
    line=dict(width=1, color='white'),
    name=''
)

outer_sphere_outline = go.Scatter3d(
    x=outer_sphere_x.flatten(),
    y=outer_sphere_y.flatten(),
    z=outer_sphere_z.flatten(),
    mode='lines',
    line=dict(width=1, color='white'),
    name=''
)


mask_inner_sphere = R < R_inner
mask_between_spheres = (R >= R_inner) & (R <= R_outer)

Ex[mask_between_spheres] = k * (q_inner) * (X[mask_between_spheres] / (R[mask_between_spheres]**3))
Ey[mask_between_spheres] = k * (q_inner) * (Y[mask_between_spheres] / (R[mask_between_spheres]**3))
Ez[mask_between_spheres] = k * (q_inner) * (Z[mask_between_spheres] / (R[mask_between_spheres]**3))


# Create an interactive 3D plot
fig = go.Figure(data=go.Cone(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), u=Ex.flatten(), v=Ey.flatten(), w=Ez.flatten(), sizemode="absolute", colorbar=dict(title="Electric Field")))
fig.update_layout(scene=dict(aspectmode="data"))
fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=1)))

# Set arrow length and scale for better visibility
arrow_length = 3.0  # Adjust the arrow length as needed
arrow_scale = 0.1  # Adjust the arrow scale as needed
fig.update_traces(u=np.array(Ex.flatten()) * arrow_length, v=np.array(Ey.flatten()) * arrow_length, w=np.array(Ez.flatten()) * arrow_length, selector=dict(type='cone'))
fig.update_traces(u=fig.data[0].u * arrow_scale, v=fig.data[0].v * arrow_scale, w=fig.data[0].w * arrow_scale, selector=dict(type='cone'))

# Set axis labels and title
fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
fig.update_layout(title='Interactive Electric Field Vectors Between Two Concentric Spherical Shells')

# Add the central charge outlines
fig.add_trace(inner_sphere_outline)
fig.add_trace(outer_sphere_outline)

# Show the interactive plot
fig.show()
