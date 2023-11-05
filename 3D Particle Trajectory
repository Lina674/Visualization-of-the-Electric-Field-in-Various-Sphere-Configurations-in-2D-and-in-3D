import numpy as np
import plotly.graph_objects as go

# Constants
k = 8.99e9  # Coulomb's constant

# Radii of the spheres
R_inner = float(input('Enter the inner radius: '))  # Inner sphere radius
R_outer = float(input('Enter the outer radius: '))  # Outer sphere radius
R_outer2 = R_outer + float(input('Enter the thickness'))

# Charge values for the inner and outer spheres

q_inner = float(input('Enter a first charge: '))  # Charge on the inner sphere (in Coulombs)
q_outer = float(input('Enter a second charge: '))  # Charge on the outer sphere (in Coulombs)


initial_position = np.array([float(input('x: ')), float(input('y: ')), float(input('z: '))], dtype=np.float64)  # Initial position of the particle
initial_velocity = np.array([float(input('vx: ')), float(input('vy: ')), float(input('vz: '))], dtype=np.float64)  # Initial velocity of the particle


# Particle properties
charge = input('Electron or Proton? ')  # Charge of the particle (in Coulombs)
mass = ''
if charge == 'Electron':
    charge = 1.6e-19
    mass = 9.1e-31
elif charge == 'Proton':
    charge = -1.6e-19
    mass = 1.6e-27     # Mass of the particle (in kg)

# Define the grid for the 3D plot
x = np.linspace(-(R_outer2 + 100), (R_outer2 + 100), 20)
y = np.linspace(-(R_outer2 + 100), (R_outer2 + 100), 20)
z = np.linspace(-(R_outer2 + 100), (R_outer2 + 100), 20)
X, Y, Z = np.meshgrid(x, y, z)

# Calculate the electric field between the spheres
R = np.sqrt(X**2 + Y**2 + Z**2)
Ex = np.zeros_like(X)
Ey = np.zeros_like(Y)
Ez = np.zeros_like(Z)


# Loop through the grid and calculate electric field
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        for k in range(X.shape[2]):
            r = R[i, j, k]

            if R_inner**3 <= r**3 <= R_outer**3:
                Ex[i, j, k] = k * q_inner * X[i, j, k] / (r**3)
                Ey[i, j, k] = k * q_inner * Y[i, j, k] / (r**3)
                Ez[i, j, k] = k * q_inner * Z[i, j, k] / (r**3)
            elif r >= R_outer2:
                Ex[i, j, k] = k * (q_inner + q_outer) * X[i, j, k] / (r**3)
                Ey[i, j, k] = k * (q_inner + q_outer) * Y[i, j, k] / (r**3)
                Ez[i, j, k] = k * (q_inner + q_outer) * Z[i, j, k] / (r**3)

# Time parameters
total_time = 1e-6  # Total simulation time (in seconds)
delta_t = 1e-9     # Time step (in seconds)
num_steps = int(total_time / delta_t)

# Arrays to store position data
positions = [initial_position]

# Simulate the motion of the charged particle
current_position = initial_position
current_velocity = initial_velocity

for _ in range(num_steps):
    # Calculate electric field at the current position using trilinear interpolation
    x_idx = np.argmin(np.abs(x[0] - current_position[0]))
    y_idx = np.argmin(np.abs(y - current_position[1]))
    z_idx = np.argmin(np.abs(z - current_position[2]))

    # Check if indices are within bounds
    x_idx = max(1, min(x.shape[0] - 2, x_idx))
    y_idx = max(1, min(y.shape[0] - 2, y_idx))
    z_idx = max(1, min(z.shape[0] - 2, z_idx))

    dx = (current_position[0] - x[x_idx]) / (x[x_idx + 1] - x[x_idx])
    dy = (current_position[1] - y[y_idx]) / (y[y_idx + 1] - y[y_idx])
    dz = (current_position[2] - z[z_idx]) / (z[z_idx + 1] - z[z_idx])


    # Perform trilinear interpolation for Ex, Ey, and Ez
    Ex_interp = (
        (1 - dx) * (1 - dy) * (1 - dz) * Ex[z_idx, y_idx, x_idx] +
        dx * (1 - dy) * (1 - dz) * Ex[z_idx, y_idx, x_idx + 1] +
        (1 - dx) * dy * (1 - dz) * Ex[z_idx, y_idx + 1, x_idx] +
        dx * dy * (1 - dz) * Ex[z_idx, y_idx + 1, x_idx + 1] +
        (1 - dx) * (1 - dy) * dz * Ex[z_idx + 1, y_idx, x_idx] +
        dx * (1 - dy) * dz * Ex[z_idx + 1, y_idx, x_idx + 1] +
        (1 - dx) * dy * dz * Ex[z_idx + 1, y_idx + 1, x_idx] +
        dx * dy * dz * Ex[z_idx + 1, y_idx + 1, x_idx + 1]
    )
    Ey_interp = (
        (1 - dx) * (1 - dy) * (1 - dz) * Ey[z_idx, y_idx, x_idx] +
        dx * (1 - dy) * (1 - dz) * Ey[z_idx, y_idx, x_idx + 1] +
        (1 - dx) * dy * (1 - dz) * Ey[z_idx, y_idx + 1, x_idx] +
        dx * dy * (1 - dz) * Ey[z_idx, y_idx + 1, x_idx + 1] +
        (1 - dx) * (1 - dy) * dz * Ey[z_idx + 1, y_idx, x_idx] +
        dx * (1 - dy) * dz * Ey[z_idx + 1, y_idx, x_idx + 1] +
        (1 - dx) * dy * dz * Ey[z_idx + 1, y_idx + 1, x_idx] +
        dx * dy * dz * Ey[z_idx + 1, y_idx + 1, x_idx + 1]
    )
    Ez_interp = (
        (1 - dx) * (1 - dy) * (1 - dz) * Ez[z_idx, y_idx, x_idx] +
        dx * (1 - dy) * (1 - dz) * Ez[z_idx, y_idx, x_idx + 1] +
        (1 - dx) * dy * (1 - dz) * Ez[z_idx, y_idx + 1, x_idx] +
        dx * dy * (1 - dz) * Ez[z_idx, y_idx + 1, x_idx + 1] +
        (1 - dx) * (1 - dy) * dz * Ez[z_idx + 1, y_idx, x_idx] +
        dx * (1 - dy) * dz * Ez[z_idx + 1, y_idx, x_idx + 1] +
        (1 - dx) * dy * dz * Ez[z_idx + 1, y_idx + 1, x_idx] +
        dx * dy * dz * Ez[z_idx + 1, y_idx + 1, x_idx + 1]
    )

    # Calculate the force experienced by the particle
    force = charge * np.array([Ex_interp, Ey_interp, Ez_interp])

    # Calculate acceleration
    acceleration = force / mass

    # Update velocity and position using the equations of motion
    current_velocity += acceleration * delta_t
    current_position = current_position.astype(float) + current_velocity * delta_t

    # Append the current position to the positions array
    positions.append(current_position)

# Create a 3D trajectory plot of the charged particle
fig = go.Figure(data=go.Scatter3d(
    x=[p[0] for p in positions],
    y=[p[1] for p in positions],
    z=[p[2] for p in positions],
    mode='lines',
    line=dict(width=5)
))

# Set axis labels and title
fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
fig.update_layout(title='Trajectory of a Charged Particle in an Electric Field')


# Show the trajectory plot
fig.show()
