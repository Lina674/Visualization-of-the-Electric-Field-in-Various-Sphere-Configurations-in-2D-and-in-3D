import numpy as np
import plotly.graph_objects as go

# Constants
k = 8.99e9  # Coulomb's constant

# Radii of the spheres
R_inner = float(input('Enter the radius of the inner sphere: '))  # Inner sphere radius
R_outer = float(input('Enter the radius of the outer shell: '))  # Outer sphere radius
R_outer2 = R_outer + float(input('Enter the thickness of the outer shell: '))

# Charge values for the inner and outer spheres
q_inner = float(input('Enter the charge of the inner sphere: '))  # Charge on the inner sphere (in Coulombs)
q_outer = float(input('Enter the charge of the outer shell: '))   # Charge on the outer sphere (in Coulombs)


# Define the grid for the 3D plot
x = np.linspace(-(R_outer2 + 50), (R_outer2 + 50), 20)
y = np.linspace(-(R_outer2 + 50), (R_outer2 + 50), 20)
z = np.linspace(-(R_outer2 + 50), (R_outer2 + 50), 20)
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

outer_sphere2_x = R_outer2 * np.sin(theta) * np.cos(phi)
outer_sphere2_y = R_outer2 * np.sin(theta) * np.sin(phi)
outer_sphere2_z = R_outer2 * np.cos(theta)

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

outer_sphere2_outline = go.Scatter3d(
    x=outer_sphere2_x.flatten(),
    y=outer_sphere2_y.flatten(),
    z=outer_sphere2_z.flatten(),
    mode='lines',
    line=dict(width=1, color='white'),
    name=''
)

# Loop through the grid and calculate electric field
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        for k in range(X.shape[2]):
            r = R[i, j, k]

            if R_inner <= r <= R_outer:
                Ex[i, j, k] = k * q_inner * X[i, j, k] / (r**3)
                Ey[i, j, k] = k * q_inner * Y[i, j, k] / (r**3)
                Ez[i, j, k] = k * q_inner * Z[i, j, k] / (r**3)
            elif r >= R_outer2:
                Ex[i, j, k] = k * (q_inner + q_outer) * X[i, j, k] / (r**3)
                Ey[i, j, k] = k * (q_inner + q_outer) * Y[i, j, k] / (r**3)
                Ez[i, j, k] = k * (q_inner + q_outer) * Z[i, j, k] / (r**3)
q
# Create an interactive 3D plot
fig = go.Figure(data=go.Cone(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), u=Ex.flatten(), v=Ey.flatten(), w=Ez.flatten(), sizemode="absolute", colorbar=dict(title="Electric Field")))
fig.update_layout(scene=dict(aspectmode="data"))
fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=1)))

# Set axis labels and title
fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
fig.update_layout(title='Interactive Electric Field Vectors Resulting from a Conducting Sphere and a Conducting Shell (No Vectors within the Conductors)')

# Add the central charge outlines
fig.add_trace(inner_sphere_outline)
fig.add_trace(outer_sphere_outline)

# Add the central charge outlines
fig.add_trace(inner_sphere_outline)
fig.add_trace(outer_sphere_outline)
fig.add_trace(outer_sphere2_outline)

# Show the interactive plot
fig.show()
