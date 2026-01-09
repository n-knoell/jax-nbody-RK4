from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, lax
import time
import numpy as np
from matplotlib import pyplot as plt

# sampler (plummer and virialized sphere)
from _nbody_sampler import plummer_sampler, virialized_sphere_sampler

jax.config.update('jax_enable_x64', True)

def acceleration(positions: jnp.ndarray, masses: jnp.ndarray, eps: float = 1e-12):
    """
    Compute accelerations for n bodies due to mutual gravity.
    positions: shape (n,3)
    masses: shape (n,)
    returns: acc shape (n,3) where acc[i] = sum_{j != i} - masses[j] * (r_i - r_j) / |r_i - r_j|^3
    (G is assumed 1)
    """
    diff = positions[:, None, :] - positions[None, :, :] 
    r2 = jnp.sum(diff ** 2, axis=-1) 
    inv_r3 = jnp.where(r2 > 0, 1.0 / (r2 * jnp.sqrt(r2) + eps), 0.0) 
    # mass-weighted factor for each pair (broadcast masses[j] over i)
    mass_factors = masses[None, :]  
    # acceleration contribution from j on i: - mass_j * diff_ij * inv_r3_ij
    contrib = - (mass_factors[..., None] * diff) * inv_r3[..., None] 
    contrib = contrib * (1.0 - jnp.eye(positions.shape[0])[:, :, None])
    acc = jnp.sum(contrib, axis=1) 
    return acc

 
@jit
def rk4_step_nbody(state: jnp.ndarray, h: float, masses: jnp.ndarray):
    """
    One RK4 step for n-body system.
    state: array shape (n,7) where each row is [t, x, y, z, vx, vy, vz]
    masses: array shape (n,)
    returns: new_state shape (n,7)
    """
    hh = 0.5 * h
    h6 = h / 6.0

    def deriv(s):
        # s shape (n,7)
        n = s.shape[0]
        dt_col = jnp.ones((n, 1), dtype=s.dtype)  # time derivative (1 per body)
        positions = s[:, 1:4]  
        velocities = s[:, 4:7] 
        acc = acceleration(positions, masses) 
        return jnp.concatenate([dt_col, velocities, acc], axis=1)  # (n,7)

    k1 = deriv(state)
    k2 = deriv(state + hh * k1)
    k3 = deriv(state + hh * k2)
    k4 = deriv(state + h * k3)

    return state + h6 * (k1 + 2.0 * (k2 + k3) + k4)


def integrate_nbody(orbits: jnp.ndarray,
                    masses: jnp.ndarray,
                    h: float,
                    T: float,
                    COM_frame: bool,
                    ):
    """
    orbits: jnp.array shape (n,7): [t, x, y, z, vx, vy, vz]
    masses: jnp.array shape (n,)
    h: timestep
    T: total integration time
    returns: traj_com shape (num_steps, n, 7) positions/velocities in COM frame
    """
    state0 = orbits  # (n,7)
    num_steps = int(jnp.ceil(T / h))
    n = state0.shape[0]
    totalM = jnp.sum(masses)

    @jit
    def run_with_fori_loop(state0):
        def body_fn(i, carry):
            state, traj = carry 
            new_state = rk4_step_nbody(state, h, masses)
            traj = traj.at[i].set(new_state)
            return new_state, traj

        traj = jnp.zeros((num_steps, n, 7), dtype=state0.dtype)
        _, traj = lax.fori_loop(0, num_steps, body_fn, (state0, traj))
        return traj

    traj = run_with_fori_loop(state0)
    # Transform trajectories into center-of-mass frame
    positions = traj[:, :, 1:4]  
    weighted = positions * masses[None, :, None]  
    COM = jnp.sum(weighted, axis=1) / totalM  #
    # Subtract COM from each body's positions for all timesteps
    positions_com = positions - COM[:, None, :] 
    traj_com = traj.at[:, :, 1:4].set(positions_com)

    if COM_frame == True:
        return traj_com
    elif COM_frame == False:
        return traj


def set_axes_equal(ax):
    """
    Make 3D axes have equal scale.
    Matplotlib 3D doesn't support `ax.set_aspect('equal')` for 3D
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_3d_trajectories(traj,
                         phase_pos=None,
                         plotname: str = "default",
                         box_size: float = 2,
                         elev: float = 30.0,
                         azim: float = 45.0,
                         show_initial: bool = True,
                         figsize=(12, 6)):
    """
    Plot 3D trajectories. Parameters
    - traj: (num_steps, n, 7) to plot
    - elev, azim: camera elevation and azimuth (degrees) -> diagonal viewpoint
    - show_initial: mark initial position
    """

    traj = np.asarray(traj) 
    num_steps, n, _ = traj.shape

    fig = plt.figure(figsize=figsize)

    def _plot_on_axis(ax, data, subtitle):
        for i in range(n):
            xs = data[:, i, 1]  
            ys = data[:, i, 2]  
            zs = data[:, i, 3]  
            ax.plot(xs, ys, zs, linewidth=1)  
            if show_initial:
                ax.scatter(xs[0], ys[0], zs[0], s=40, label="Initial")  # initial point
            if phase_pos is not None:
                x_phase, y_phase, z_phase = phase_pos[i]
                ax.scatter(x_phase, y_phase, z_phase, s=40, label="Phase") # phase point
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(subtitle)
        ax.view_init(elev=elev, azim=azim)
        
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
        set_axes_equal(ax)

  
    ax = fig.add_subplot(111, projection='3d')
    _plot_on_axis(ax, traj, "n-body trajectories (n="+str(n)+")")
    size = box_size
    ax.set_xlim3d([-size, size])
    ax.set_ylim3d([-size, size])
    ax.set_zlim3d([-size, size])
    plt.savefig(plotname+"_3d.png", dpi=300)

def plot2d(masses, traj_com, plotname, phase_pos=None):
    for i in range(len(masses)):
        plt.plot(traj_com[:, i, 1], traj_com[:, i, 2], markersize=.1, linewidth=1)
        if phase_pos is not None:
            x_phase, y_phase, z_phase = phase_pos[i]
            plt.scatter(x_phase, y_phase, s=40, label="Phase") # phase point
    plt.axis('equal')
    plt.xlim(-box_size,box_size)
    plt.ylim(-box_size,box_size)
    plt.legend(loc='upper left')
    plt.title("n-body (COM frame), "+str(N)+" objects")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(plotname+".png", dpi=300)
    plt.close()
    
def orbital_circ_velocity(mass, radius):
    """Compute circular orbital velocity for given mass and radius."""
    v = jnp.sqrt(mass/radius)
    return v

def time_to_integrate(orbits, masses, h, T, COM_frame):
    """compute time to integrate n-body system"""
    t0 = time.perf_counter()
    traj_com = integrate_nbody(orbits, masses, h, T, COM_frame)  
    t1 = time.perf_counter()
    return t1 - t0


@jit
def binary_starting_orbits(sep: float,
                           e: float,
                           inc_deg: float,
                           m1: float,
                           m2: float,
                           true_anom_deg: float = 0.0,
                           G: float = 1.0):
    """
    Construct initial state vectors [t, x, y, z, vx, vy, vz] for a binary system
    - sep: semi-major axis a
    - e: eccentricity (0 <= e < 1 for elliptic)
    - inc_deg: inclination in degrees (rotation about x-axis). Position stays on x-axis.
    - m1, m2: masses
    - true_anom_deg: starting true anomaly (degrees); default = 0 -> periapsis (relative position on +x)
    - G: gravitational constant (default 1)
    Returns: jnp.array shape (2,7) where each row is [t, x, y, z, vx, vy, vz]
    Notes:
      - Positions are given in the center-of-mass frame (COM at origin).
      - By construction initial positions have only an x-component (y=z=0).
    """

    # convert angles
    i = jnp.deg2rad(inc_deg)
    f = jnp.deg2rad(true_anom_deg)

    # semi-major axis
    a = sep

    mu = G * (m1 + m2)

    # radial distance at true anomaly f
    r = a * (1.0 - e**2) / (1.0 + e * jnp.cos(f))

    # specific angular momentum
    h = jnp.sqrt(mu * a * (1.0 - e**2))

    # velocity components in perifocal (r, theta, z=0)
    v_r = (mu / h) * e * jnp.sin(f)
    v_theta = (mu / h) * (1.0 + e * jnp.cos(f))

    # position and velocity in perifocal coords (relative vector)
    r_pf = jnp.array([r, 0.0, 0.0], dtype=jnp.float64)
    v_pf = jnp.array([v_r, v_theta, 0.0], dtype=jnp.float64)

    # rotation about x-axis by inclination i (perifocal -> inertial, with Omega=omega=0)
    ci = jnp.cos(i)
    si = jnp.sin(i)
    R_x = jnp.array([[1.0, 0.0, 0.0],
                     [0.0, ci, -si],
                     [0.0, si,  ci]], dtype=jnp.float64)

    r_eci = R_x @ r_pf
    v_eci = R_x @ v_pf

    # split into two-body COM coordinates (COM at origin)
    r1 = - (m2 / (m1 + m2)) * r_eci
    r2 =   (m1 / (m1 + m2)) * r_eci
    v1 = - (m2 / (m1 + m2)) * v_eci
    v2 =   (m1 / (m1 + m2)) * v_eci

    orbit1 = jnp.concatenate([jnp.array([0.0]), r1, v1])  # [t, x, y, z, vx, vy, vz]
    orbit2 = jnp.concatenate([jnp.array([0.0]), r2, v2])

    return jnp.stack([orbit1, orbit2])


@jit
def _eccentric_from_true_anomaly(f, e):
    """
    E = 2*arctan( sqrt((1-e)/(1+e)) * tan(f/2) )
    Handles f near ±pi robustly via atan2 formulation.
    """
    # Use half-angle formulation with atan2 to reduce issues with quadrants.
    s = jnp.sqrt((1.0 - e) / (1.0 + e))
    t_half = jnp.tan(0.5 * f)
    E = 2.0 * jnp.arctan(s * t_half)
    # Ensure E is continuous (map to principal branch)
    return E

@jit
def _solve_kepler_newton(M, e, n_iter=30):
    """
    Solve M = E - e*sin(E) for E using Newton iterations.
    M, e are scalars. Uses a good analytic initial guess.
    """
    # initial guess using Fourier series / first terms
    E0 = M + e * jnp.sin(M) + 0.5 * (e**2) * jnp.sin(2.0 * M)

    def body_fun(i, E):
        f = E - e * jnp.sin(E) - M
        fp = 1.0 - e * jnp.cos(E)
        E_new = E - f / fp
        return E_new

    E_final = lax.fori_loop(0, n_iter, body_fun, E0)
    return E_final

@jit
def binary_starting_orbits_at_phase(sep: float,
                           e: float,
                           inc_deg: float,
                           m1: float,
                           m2: float,
                           phi: float = 0.0,
                           true_anom_deg: float = 0.0,
                           G: float = 1.0):
    """
    Construct state vectors [t, x, y, z, vx, vy, vz] for a binary at orbital phase `phi`.
    - sep: semi-major axis a
    - e: eccentricity (0 <= e < 1 for elliptic)
    - inc_deg: inclination in degrees (rotation about x-axis). Positions are rotated about x.
    - m1, m2: masses
    - phi: orbital phase in [0,1). phi=0 corresponds to true_anom_deg at time zero.
    - true_anom_deg: true anomaly at phase phi=0 (degrees). Default 0 -> periapsis on +x.
    - G: gravitational constant (default 1)
    Returns: jnp.array shape (2,7) where each row is [t, x, y, z, vx, vy, vz]
    Notes:
      - Positions/velocities are returned in the center-of-mass frame (COM at origin).
      - By construction the initial reference periapsis (phi=0, true_anom_deg=0) lies on +x.
    """

    # convert angles and scalars
    i = jnp.deg2rad(inc_deg)
    f0 = jnp.deg2rad(true_anom_deg)
    a = sep
    mu = G * (m1 + m2)

    # 1) compute eccentric anomaly at phi=0 from provided true anomaly f0
    # handle circular case e==0 specially to avoid divisions by zero
    E0 = jnp.where(e == 0.0, f0, _eccentric_from_true_anomaly(f0, e))

    # 2) compute mean anomaly at phi=0
    M0 = E0 - e * jnp.sin(E0)

    # 3) advance mean anomaly by 2*pi*phi (wrap phi into [0,1))
    phi_wrap = phi - jnp.floor(phi)
    M_target = M0 + 2.0 * jnp.pi * phi_wrap

    # 4) solve Kepler's equation for target eccentric anomaly E_target
    E = _solve_kepler_newton(M_target, e, n_iter=40)

    # 5) compute position in perifocal coordinates from E
    cosE = jnp.cos(E)
    sinE = jnp.sin(E)
    sqrt_1_e2 = jnp.sqrt(jnp.maximum(0.0, 1.0 - e**2))

    # Perifocal position (relative) (x_pf, y_pf, 0)
    x_pf = a * (cosE - e)
    y_pf = a * sqrt_1_e2 * sinE
    r_pf = jnp.array([x_pf, y_pf, 0.0], dtype=jnp.float64)

    # Perifocal velocity (relative)
    # factor = sqrt(mu / a) / (1 - e*cosE)
    factor = jnp.sqrt(mu / a) / (1.0 - e * cosE)
    vx_pf = - factor * sinE
    vy_pf =   factor * sqrt_1_e2 * cosE
    v_pf = jnp.array([vx_pf, vy_pf, 0.0], dtype=jnp.float64)

    # 6) rotate by inclination about x-axis (perifocal -> inertial, with Omega=omega=0)
    ci = jnp.cos(i)
    si = jnp.sin(i)
    R_x = jnp.array([[1.0, 0.0, 0.0],
                     [0.0, ci, -si],
                     [0.0, si,  ci]], dtype=jnp.float64)

    r_eci = R_x @ r_pf
    v_eci = R_x @ v_pf

    # 7) split into two-body COM coordinates (COM at origin)
    r1 = - (m2 / (m1 + m2)) * r_eci
    r2 =   (m1 / (m1 + m2)) * r_eci
    v1 = - (m2 / (m1 + m2)) * v_eci
    v2 =   (m1 / (m1 + m2)) * v_eci

    orbit1 = jnp.concatenate([jnp.array([0.0]), r1, v1])  # [t, x, y, z, vx, vy, vz]
    orbit2 = jnp.concatenate([jnp.array([0.0]), r2, v2])

    return jnp.stack([orbit1, orbit2])


@jit
def kepler_period(a: float, m1: float, m2: float, G: float = 1.0):
    return 2.0 * jnp.pi * jnp.sqrt(a**3 / (G * (m1 + m2)))

@jit
def positions_at_phase_kepler(traj: jnp.ndarray,
                              phi: float,
                              a: float,
                              m1: float,
                              m2: float,
                              G: float = 1.0):
    """
    Return positions (x,y,z) of both stars at orbital phase phi (0..1)
    using analytic Kepler period.
    """
    times = traj[:, 0, 0]
    period = kepler_period(a, m1, m2, G)

    phi = phi - jnp.floor(phi)  # wrap
    target_time = times[0] + phi * period

    j = jnp.searchsorted(times, target_time, side="right") - 1
    j = jnp.clip(j, 0, traj.shape[0] - 2)

    t0 = times[j]
    t1 = times[j + 1]
    alpha = (target_time - t0) / (t1 - t0)

    # positions
    pos0 = traj[j, :, 1:4]
    pos1 = traj[j + 1, :, 1:4]

    # velocities
    vel0 = traj[j, :, 4:7]
    vel1 = traj[j + 1, :, 4:7]

    pos_phi = (1.0 - alpha) * pos0 + alpha * pos1
    vel_phi = (1.0 - alpha) * vel0 + alpha * vel1

    return pos_phi, vel_phi


if __name__ == "__main__":
    #### circular orbits in one plane (xy)
    # M = 1.0
    # r1=3.8
    # r2=1
    # r3=1.9
    # r4 =2.7
    # orbit1 = jnp.array([0.0, 0., 0.0, 0.0, 0.0, 0., 0.0])
    # orbit2 = jnp.array([0.0, r1, 0.0, 0.0, 0.0,orbital_circ_velocity(M,r1), 0.0])
    # orbit3 = jnp.array([0.0, 0, r2, 0, orbital_circ_velocity(M,r2) ,-0.0, 0.0])
    # orbit4 = jnp.array([0.0, 0, -r3, 0.0, -orbital_circ_velocity(M,r3),-0.0, 0.0])
    # orbit5 = jnp.array([0.0, -r4, 0, 0, 0.0,-orbital_circ_velocity(M,r4), 0.0])
    # orbits = jnp.stack([orbit1, orbit2, orbit3, orbit4, orbit5]) 
    # masses = jnp.array([1, 0.95, 0.98, 1.02, 1.03])*M

    ### binary system time test
    # orbit1 = np.array([0.0,  0.5, 0.0, 0.0, 0.0, 0.5, 0.0])
    # orbit2 = np.array([0.0, -0.5, 0.0, 0.0, 0.0,-0.5, 0.0])
    # orbits = jnp.stack([orbit1, orbit2])
    # M1 = M2 = 0.5
    # h = 0.001
    # masses = jnp.array([M1, M2])
    # T = 500

    # Tmax=500
    # T_array = np.arange(1, Tmax, 50)
    # time_taken_list = []
    # for i in T_array:
    #     time_taken = time_to_integrate(orbits, masses, h, i, COM_frame=True)
    #     time_taken_list.append(time_taken)
    #     print(i, time_taken)

    # np.savez_compressed('JAX_integration_times_binary.npz', T_array=T_array, time_taken_list=time_taken_list)
    
    h = 0.001
    T = 500

    N=2
    box_size=0.5
    mass_source = 1e-3
    low_mass = 0.1*mass_source
    high_mass =10*low_mass
    # orbits, masses = virialized_sphere_sampler(N, low_mass, high_mass, a=box_size/2, seed=0)
    orbits, masses = plummer_sampler(N,key=jax.random.PRNGKey(1), M1=low_mass, M2=high_mass, a=box_size/2)
    COM_frame = True

    mass_source1 =  1
    mass_source2 = mass_source1 * 1.5
    R_forced = 3.0
    v_circ = orbital_circ_velocity(mass_source1+mass_source2, R_forced)
    v_circ = jnp.sqrt((mass_source1 + mass_source2) / R_forced)
    v_1 = mass_source2 / (mass_source1 + mass_source2) * v_circ
    v_2 = mass_source1 / (mass_source1 + mass_source2) * v_circ
    orbit1 = jnp.array([0.0, R_forced/2, 0.0, 0.0, 0.0, -v_1, 0.0])    #this is clockwise rotation in the xy plane
    orbit2 = jnp.array([0.0, -R_forced/2, 0.0, 0.0, 0.0, v_2, 0.0])
    orbits = jnp.stack([orbit1, orbit2])
    h = 0.001
    masses = jnp.array([mass_source1, mass_source2])

    # example
    a = 0.34        # semi-major axis
    e = 0.8
    inc = 30    # degrees
    mass_source1 = 1.0e-4
    mass_source2 = 1.5e-4

    Period = 2 * jnp.pi * jnp.sqrt((a**3) / (mass_source1 + mass_source2))

    phase = 0.5  # x of an orbit past reference time
    T = Period * phase

    orbits = binary_starting_orbits_at_phase(a, e, inc, mass_source1, mass_source2, 0.5, true_anom_deg=0.0)
    # orbits = binary_starting_orbits(a, e, inc, mass_source1, mass_source2, true_anom_deg=0.0)
    print("Initial orbits:\n", orbits)
    masses = jnp.array([mass_source1, mass_source2])
    
    traj_com = integrate_nbody(orbits, masses, h, T, COM_frame)  

    phase_pos, phase_vel = positions_at_phase_kepler(traj_com, phase, a, masses[0], masses[1])   # returns array (2,3)

    x1, y1, z1 = phase_pos[0]
    vx1, vy1, vz1 = phase_vel[0]
    x2, y2, z2 = phase_pos[1]
    vx2, vy2, vz2 = phase_vel[1]
    
    plotname = "test"   #+str(N)
    plot = "both"     # "2d", "3d", "both"
    if plot == "2d":
        plot2d(masses, traj_com, plotname)  
    elif plot == "3d":
        plot_3d_trajectories(traj_com, plotname, box_size, elev=35, azim=45)
    elif plot == "both":
        plot2d(masses, traj_com, plotname, phase_pos)  
        plot_3d_trajectories(traj_com, phase_pos, plotname, box_size, elev=35, azim=45)
