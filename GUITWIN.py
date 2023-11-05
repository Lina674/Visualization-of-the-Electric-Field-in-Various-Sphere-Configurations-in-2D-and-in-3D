import PySimpleGUI as sg
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import numpy as np
from numpy import linspace
import plotly.graph_objects as go

def projectile (R_inner, R_outer, R_outer2, q_inner, q_outer, x, y, z, vx, vy , vz, charger):

    # Constants
    k = 8.99e9  # Coulomb's constant

    # Radii of the spheres
    

    initial_position = np.array([x, y, z], dtype=np.float64)  # Initial position of the particle
    initial_velocity = np.array([vx, vy, vz], dtype=np.float64)  # Initial velocity of the particle


    # Particle properties
     # Charge of the particle (in Coulombs)
    mass = ''
    if charger == 'E':
        charger = 1.6e-19
        mass = 9.1e-31
    elif charger == 'P':
        charger = -1.6e-19
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
        force = charger * np.array([Ex_interp, Ey_interp, Ez_interp])

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



# Define the calculation function for Sphere with conductor (electric field and potential)
def electric_field_lines(ra, ch):
    k = 8.99e9
    x = np.linspace(-(2*ra), (2*ra), 20)
    y = np.linspace(-(2*ra), (2*ra), 20)
    z = np.linspace(-(2*ra), (2*ra), 20)
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
    inner_sphere_x = ra * np.sin(theta) * np.cos(phi)
    inner_sphere_y = ra * np.sin(theta) * np.sin(phi)
    inner_sphere_z = ra * np.cos(theta)

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
    outside_sphere = (R >= ra)
    Ex[outside_sphere] = k * ch * X[outside_sphere] / (R[outside_sphere]**3 + 1e-10)  # Added a small value to avoid division by zero
    Ey[outside_sphere] = k * ch * Y[outside_sphere] / (R[outside_sphere]**3 + 1e-10)
    Ez[outside_sphere] = k * ch * Z[outside_sphere] / (R[outside_sphere]**3 + 1e-10)

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

def co_electric_field_lines(radin, radout, chin):
        # Constants
    k = 8.99e9  # Coulomb's constant
    # Define the grid for the 3D plot
    x = np.linspace(-radout, radout, 20)
    y = np.linspace(-radout, radout, 20)
    z = np.linspace(-radout, radout, 20)
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
    inner_sphere_x = radin * np.sin(theta) * np.cos(phi)
    inner_sphere_y = radin * np.sin(theta) * np.sin(phi)
    inner_sphere_z = radin * np.cos(theta)

    outer_sphere_x = radout * np.sin(theta) * np.cos(phi)
    outer_sphere_y = radout * np.sin(theta) * np.sin(phi)
    outer_sphere_z = radout * np.cos(theta)

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


    mask_inner_sphere = R < radin
    mask_between_spheres = (R >= radin) & (R <= radout)

    Ex[mask_between_spheres] = k * (chin) * (X[mask_between_spheres] / (R[mask_between_spheres]**3))
    Ey[mask_between_spheres] = k * (chin) * (Y[mask_between_spheres] / (R[mask_between_spheres]**3))
    Ez[mask_between_spheres] = k * (chin) * (Z[mask_between_spheres] / (R[mask_between_spheres]**3))


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

def calculate_sphere_conductor(radius, charge):
    xList = linspace(0, 2 * radius, 5000)
    yList = [(9 * 10**9) * charge / x**2 if x > radius else 0 for x in xList]

    # Calculate electric potential for conductor
    vList = linspace(0, 2 * radius, 5000)
    zList = []
    for v in vList:
        if v <= radius:
            zList.append((9 * 10**9) * charge / radius)
        elif v <= 7 * radius:
            zList.append((9 * 10**9) * charge / v)
        else:
            zList.append(0)

    # Plot electric field and electric potential
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(xList, yList)
    plt.xlabel("Distance from the Center (m)")
    plt.ylabel("Electric Field (N/C)")
    plt.title("Electric Field as a Function of Distance")

    plt.subplot(122)
    plt.plot(vList, zList)
    plt.xlabel("Distance from the Center (m)")
    plt.ylabel("Electric Potential (V)")
    plt.title("Electric Potential as a Function of Distance")

    plt.tight_layout()
    plt.show()

# Define the calculation for a hollow conductor
def calculate_hollow_sphere_conductor(inner_rad, outer_rad, hcharge):
    xList = linspace(0, 2 * radius, 5000)
    yList = []
    for x in xList:
        if x <= inner_rad:
            yList.append(0)
        elif x <= 7 * outer_rad:
            yList.append(0)
        else:
            yList.append((9 * 10**9) * hcharge / (x**2))

# Plot electric field and electric potential
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(xList, yList)
    plt.xlabel("Distance from the Center (m)")
    plt.ylabel("Electric Field (N/C)")
    plt.title("Electric Field as a Function of Distance")

    plt.tight_layout()
    plt.show()


# Define the calculation for a hollow insulator
def calculate_hollow_sphere_insulator(inner_rad, outer_rad, hcharge):
    xList = linspace(0, 2 * radius, 5000)
    yList = []
    for x in xList:
        if x <= inner_rad:
            yList.append(0)
        elif x <= 7 * outer_rad:
            yList.append(((9*(10**9))*hcharge*(x**3-inner_rad**3)*(((x**2)*(outer_rad**3-inner_rad**3))**-1)))
        else:
            yList.append((9*(10**9))*hcharge*(x**-2))

# Plot electric field and electric potential
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(xList, yList)
    plt.xlabel("Distance from the Center (m)")
    plt.ylabel("Electric Field (N/C)")
    plt.title("Electric Field as a Function of Distance")

    plt.tight_layout()
    plt.show()

# Define the calculation for a sphere with an insulator (E&V)
def calculate_sphere_insulator(radius, charge):
    xList = linspace(0, 2 * radius, 5000)
    yList = []
    for x in xList:
        if x <= radius:
            yList.append((9 * 10**9) * charge * x / (radius**3))
        elif x <= 7 * radius:
            yList.append((9 * 10**9) * charge / (x**2))
        else:
            yList.append(0)

    vList = linspace(0, 2 * radius, 5000)
    zList = []
    for v in vList:
        if v <= radius:
            zList.append((9 * 10**9) * charge * (-v**2) / (2 * radius**3) + 3 * 0.5 * (9 * 10**9) * charge / radius)
        elif v <= 7 * radius:
            zList.append((9 * 10**9) * charge / v)
        else:
            zList.append(0)

    # Plot electric field and electric potential
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(xList, yList)
    plt.xlabel("Distance from the Center (m)")
    plt.ylabel("Electric Field (N/C)")
    plt.title("Electric Field as a Function of Distance")

    plt.subplot(122)
    plt.plot(vList, zList)
    plt.xlabel("Distance from the Center (m)")
    plt.ylabel("Electric Potential (V)")
    plt.title("Electric Potential as a Function of Distance")

    plt.tight_layout()
    plt.show()


# Define the calculations for a concentric outer conductor, and inner insulator
def calculate_concentric_conductor_insulator(inner_radius, inner_charge, inter_radius, outer_radius, outer_charge):
    # Check that the inner shell is an insulator and the outer shell is a conductor
    if values.get("inner_insulator") and values.get("outer_conductor"):
        # Electric field calculation
        def electric_field(x):
            if x <= inner_radius:
                return (9 * (10**9)) * inner_charge * x * (inner_radius**-3)
            if x <= inter_radius:
                return (9 * (10**9)) * inner_charge * (x**-2)
            if x <= outer_radius:
                return 0
            if x <= 3*outer_radius:
                return (9 * (10**9)) * (inner_charge + outer_charge) *(x**-2)

        xList = linspace(0, 9999, 5000)
        yList = [electric_field(x) for x in xList]

        # Electric potential calculation
        def electric_potential(v):
            if v <= inner_radius:
                return ((9 * (10**9)) * inner_charge * (-(v**2)) * (0.5) * (inner_radius**-3)) + (3 * 0.5 * (9 * (10**9)) * inner_charge * (inner_radius**-1))
            if v <= outer_radius:
                return (9 * (10**9)) * inner_charge * (v**-1)
            if v <= 3 * outer_radius:
                return (9 * (10**9)) * (inner_charge + outer_charge) * (v**-1)
            return 0

        vList = linspace(0, 9999, 5000)
        zList = [electric_potential(v) for v in vList]

        # Plot electric field and electric potential
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.plot(xList, yList)
        plt.xlabel("Distance from the Center (m)")
        plt.ylabel("Electric Field (N/C)")
        plt.title("Electric Field as a Function of Distance")

        """plt.subplot(122)
        plt.plot(vList, zList)
        plt.xlabel("Distance from the Center (m)")
        plt.ylabel("Electric Potential (V)")
        plt.title("Electric Potential as a Function of Distance")
        """
        plt.tight_layout()
        plt.show()
    

# Define the calculation function for Concentric Spheres with conductor inner and conductor outer shells (electric field and potential)
def calculate_concentric_conductor_conductor(ra, qa, rc, rb, qb):
    
    def fn(x):
        if x <= ra:
            return (0)
        if x <= rb:
            return ((9 * (10**9)) * qa * (x**-2))
        if x <= rc:
            return(0)
        if x <= 3*rc:
            return((9 * (10**9)) * (qa + qb)*(x**-2))
    xList = linspace(0, 3 * rc, 5000)
    yList = [fn(x) for x in xList]

    vList = linspace(0, 3 * rb, 5000)
    zList = []
    for v in vList:
        if v <= ra:
            zList.append(0)
        elif v <= rb:
            zList.append((9 * (10**9)) * qa * (ra - v) / (v * ra))
        else:
            zList.append((9 * (10**9)) * (qa - qb) / v)

    # Plot electric field and electric potential
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(xList, yList)
    plt.xlabel("Distance from the Center (m)")
    plt.ylabel("Electric Field (N/C)")
    plt.title("Electric Field as a Function of Distance")
    """
    plt.subplot(122)
    plt.plot(vList, zList)
    plt.xlabel("Distance from the Center (m)")
    plt.ylabel("Electric Potential (V)")
    plt.title("Electric Potential as a Function of Distance")
    """
    plt.tight_layout()
    plt.show()

def calculate_concentric_conductor_conductor(radius_inner, charge_inner, radius_inter, radius_outer, charge_outer):
    
    def fn(x):
        if x <= radius_inner:
            return 0
        if x <= radius_inter:
            return ((9 * 10**9) * charge_inner * (x**-2))
        if x <= radius_outer:
            return (0)
        if (x<=3*radius_outer):
            return ((9 * 10**9) * (charge_inner + charge_outer)*(x**-2))
    xList = linspace(0, 3 * radius_outer, 5000)
    yList = [fn(x) for x in xList]

    vList = linspace(0, 3 * radius_outer, 5000)
    zList = []
    for v in vList:
        if v <= radius_inner:
            zList.append(0)
        elif v <= radius_outer:
            zList.append((9 * 10**9) * charge_inner * (radius_inner - v) / (v * radius_inner))
        else:
            zList.append((9 * 10**9) * (charge_inner - charge_outer) / v)

    # Plot electric field and electric potential
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(xList, yList)
    plt.xlabel("Distance from the Center (m)")
    plt.ylabel("Electric Field (N/C)")
    plt.title("Electric Field as a Function of Distance")
    """
    plt.subplot(122)
    plt.plot(vList, zList)
    plt.xlabel("Distance from the Center (m)")
    plt.ylabel("Electric Potential (V)")
    plt.title("Electric Potential as a Function of Distance")
    """
    plt.tight_layout()
    plt.show()


# Layout for the main window with tabs
layout = [
    [sg.TabGroup([
        [
            sg.Tab("Sphere", [
                [sg.Text("Conductor or Insulator:")],
                [sg.Radio("Conductor", "material", key="conductor"), sg.Radio("Insulator", "material", key="insulator")],
                [sg.Text("Enter Radius (m):"), sg.InputText(key="radius")],
                [sg.Text("Enter Charge (C):"), sg.InputText(key="charge")],
                [sg.Button("Calculate Sphere")]
            ]),
            sg.Tab("Concentric Sphere", [
                [sg.Text("Inner Shell (Conductor or Insulator):")],
                [sg.Radio("Conductor", "inner_material", key="inner_conductor"), sg.Radio("Insulator", "inner_material", key="inner_insulator")],
                [sg.Text("Outer Shell (Conductor or Insulator):")],
                [sg.Radio("Conductor", "outer_material", key="outer_conductor"), sg.Radio("Insulator", "outer_material", key="outer_insulator")],
                [sg.Text("Enter Inner Shell Radius (m):"), sg.InputText(key="inner_radius")],
                [sg.Text("Enter Inter Shell Radius (m):"), sg.InputText(key="inter_radius")],
                [sg.Text("Enter Outer Shell Radius (m):"), sg.InputText(key="outer_radius")],
                [sg.Text("Enter Outer Shell Charge (C):"), sg.InputText(key="outer_charge")],
                [sg.Text("Enter Inner Shell Charge (C):"), sg.InputText(key="inner_charge")],

                [sg.Button("Calculate Concentric Sphere")]
            ]),
            sg.Tab("Hollow Sphere", [
                [sg.Text("Conductor or Insulator:")],
                [sg.Radio("Conductor", "material", key="conductor"), sg.Radio("Insulator", "material", key="insulator")],
                [sg.Text("Enter Inner Radius (m):"), sg.InputText(key="radius")],
                [sg.Text("Enter Outer Radius (m):"), sg.InputText(key="radius")],
                [sg.Text("Enter Charge (C):"), sg.InputText(key="charge")],
                [sg.Button("Calculate Hollow Sphere")]
            ]),
            sg.Tab("Electric Field 3D Lines - Sphere", [
                [sg.Text("Enter Radius (m):"), sg.InputText(key="ra")],
                [sg.Text("Enter Charge (C):"), sg.InputText(key="ch")],
                [sg.Button("Calculate 3D Lines for Sphere")]
            ]),
            sg.Tab("Electric Field 3D Lines - Concentric Spheres", [
                [sg.Text("Enter Inner Radius (m):"), sg.InputText(key="radin")],
                [sg.Text("Enter Outer Radius (m):"), sg.InputText(key="radout")],
                [sg.Text("Enter Inner Charge (C):"), sg.InputText(key="chin")],
                [sg.Button("Calculate 3D Lines for Concentric Spheres")]
            ]),
            sg.Tab("Charged Particle Trajectory", [
                [sg.Text("Enter Inner Radius (m):"), sg.InputText(key="R_inner")],
                [sg.Text("Enter Outer Radius (m):"), sg.InputText(key="R_outer")],
                [sg.Text("Enter Outer Radius 2 (m):"), sg.InputText(key="R_outer2")],
                [sg.Text("Enter Inner Charge (C):"), sg.InputText(key="q_inner")],
                [sg.Text("Enter Outer Charge (C):"), sg.InputText(key="q_outer")],
                [sg.Text("Position in x:"), sg.InputText(key="x")],
                [sg.Text("Position in y:"), sg.InputText(key="y")],
                [sg.Text("Position in z:"), sg.InputText(key="z")],
                [sg.Text("Velocity in x:"), sg.InputText(key="vx")],
                [sg.Text("Velocity in y:"), sg.InputText(key="vy")],
                [sg.Text("Velocity in z:"), sg.InputText(key="vz")],
                [sg.Text("Electron or Proton (E/P):"), sg.InputText(key="charger")],
                [sg.Button("Calculate Trajectory of Charged Particle")]
            ]),

        ]
    ])],
]

window = sg.Window("Visualization of the Electric Field in Various Sphere Configurations in 2D and 3D", layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break

    if event=="Calculate Trajectory of Charged Particle":
        R_inner = float(values.get("R_inner"))
        R_outer = float(values.get("R_outer"))
        R_outer2 = float(values.get("R_outer2"))
        q_inner = float(values.get("q_inner"))
        q_outer = float(values.get("q_outer"))
        x = float(values.get("x"))
        y = float(values.get("y"))
        z = float(values.get("z"))
        vx = float(values.get("vx"))
        vy = float(values.get("vy"))
        vz = float(values.get("vz"))
        charger =(values.get("charger"))
        projectile(R_inner, R_outer, R_outer2, q_inner, q_outer, x, y, z, vx, vy , vz, charger)
    if event=="Calculate 3D Lines for Sphere":
        ra = float(values.get("ra"))
        ch = float(values.get("ch"))
        electric_field_lines(ra, ch)
        #except ValueError:
            #sg.popup_error("Please enter valid numeric values for Radius and Charge."

    if event == "Calculate 3D Lines for Concentric Spheres":
        radin = float(values.get("radin"))
        radout = float(values.get("radout"))
        chin = float(values.get("chin"))
        co_electric_field_lines(radin, radout, chin)


    if event == "Calculate Sphere":
        if values.get("conductor"):
            try:
                radius = float(values.get("radius"))
                charge = float(values.get("charge"))
                calculate_sphere_conductor(radius, charge)
            except ValueError:
                sg.popup_error("Please enter valid numeric values for Radius and Charge.")
        elif values.get("insulator"):
            try:
                radius = float(values.get("radius"))
                charge = float(values.get("charge"))
                calculate_sphere_insulator(radius, charge)
            except ValueError:
                sg.popup_error("Please enter valid numeric values for Radius and Charge.")

    if event == "Calculate Hollow Sphere":
        if values.get("conductor"):
            try:
                hcharge = float(values.get("hcharge"))
                inner_rad = float(values.get("inner_rad"))
                outer_rad = float(values.get("outer_rad"))
                calculate_hollow_sphere_conductor(inner_rad, outer_rad, hcharge)
            except ValueError:
                sg.popup_error("Please enter valid numeric values for Radius and Charge.")
        elif values.get("insulator"):
            try:
                hcharge = float(values.get("hcharge"))
                inner_rad = float(values.get("inner_rad"))
                outer_rad = float(values.get("outer_rad"))
                calculate_hollow_sphere_insulator(inner_rad, outer_rad, hcharge)
            except ValueError:
                sg.popup_error("Please enter valid numeric values for Radius and Charge.")

    if event == "Calculate Concentric Spheres":
        if values.get("inner_conductor") and values.get("outer_conductor"):
        # Calculate and display results for Concentric Spheres with Conductor inner and outer shells
            pass
        elif values.get("inner_conductor") and values.get("outer_insulator"):
        # Calculate and display results for Concentric Spheres with Conductor inner and Insulator outer shells
            pass
        else:
            calculate_concentric_conductor_insulator(
                float(values.get("inner_radius")),
                float(values.get("inner_charge")),
                float(values.get("inter_radius")),
                float(values.get("outer_radius")),
                float(values.get("outer_charge"))
            )
            calculate_concentric_conductor_conductor(
                float(values.get("inner_radius")),
                float(values.get("inner_charge")),
                float(values.get("inter_radius")),
                float(values.get("outer_radius")),
                float(values.get("outer_charge"))    
            )

window.close()
