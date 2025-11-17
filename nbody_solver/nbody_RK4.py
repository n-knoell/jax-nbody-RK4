from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, lax
import time

# sampler
from _nbody_sampler import plummer_sampler, virialized_sphere_sampler

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
        positions = s[:, 1:4]  # (n,3)
        velocities = s[:, 4:7]  # (n,3)
        acc = acceleration(positions, masses)  # (n,3)
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
                    eps: float = 1e-12):
    """
    orbits: jnp.array shape (n,7) initial rows [t, x, y, z, vx, vy, vz]
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
            state, traj = carry  # state (n,7), traj (num_steps, n, 7)
            new_state = rk4_step_nbody(state, h, masses)
            traj = traj.at[i].set(new_state)
            return new_state, traj

        traj = jnp.zeros((num_steps, n, 7), dtype=state0.dtype)
        _, traj = lax.fori_loop(0, num_steps, body_fn, (state0, traj))
        return traj

    t0 = time.perf_counter()
    traj = run_with_fori_loop(state0)
    t1 = time.perf_counter()
    print(f"n-body RK4 took {t1 - t0:.4f} seconds")

    # Transform trajectories into center-of-mass frame (positions only; times/vels adjusted)
    positions = traj[:, :, 1:4]  
    weighted = positions * masses[None, :, None]  
    COM = jnp.sum(weighted, axis=1) / totalM  #
    # Subtract COM from each body's positions for all timesteps
    positions_com = positions - COM[:, None, :]  # (num_steps, n, 3)
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
            xs = data[:, i, 1]  # x
            ys = data[:, i, 2]  # y
            zs = data[:, i, 3]  # z
            ax.plot(xs, ys, zs, linewidth=1)  #, label=f'body {i}')
            if show_initial:
                ax.scatter(xs[0], ys[0], zs[0], s=40)  # initial point
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(subtitle)
        ax.view_init(elev=elev, azim=azim)
        
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
        set_axes_equal(ax)

  
    ax = fig.add_subplot(111, projection='3d')
    _plot_on_axis(ax, traj, "n-body trajectories (n="+str(N)+")")
    size = box_size
    ax.set_xlim3d([-size, size])
    ax.set_ylim3d([-size, size])
    ax.set_zlim3d([-size, size])
    plt.savefig(plotname+"_3d.png", dpi=300)
    
def orbital_circ_velocity(mass, radius):
    v = jnp.sqrt(mass/radius)
    return v

if __name__ == "__main__":
    jax.config.update('jax_enable_x64', True)
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    # orbit1 = jnp.array([0.0, 0.3, 0.0, 0.0, 0.0, 0.6, 0.0])
    # orbit2 = jnp.array([0.0, -0.7, 0.0, 0.0, 0.0,-0.5, 0.0])
    # orbit3 = jnp.array([0.0, 0, 0, -2, 0.5**0.5,-0.0, 0.0])
    # orbit4 = jnp.array([0.0, 0, 2.8, 0.0, (1/2.8)**0.5,-0.0, 0.0])
    # orbit5 = jnp.array([0.0, 0, 0, 1, 0.0,-1, 0.0])
    # masses = jnp.array([0.5, 0.5, 0.001, 0.0003, 0.0002])  # (2,)
    mass_source = 1e-3
    orbit1 = jnp.array([0.0,  0.991448574176414,  0.029880705565651,  0.010875687404770,  0.084207016573161,  0.022741169835371,  0.000000000000000])
    orbit2 = jnp.array([0.0,  0.991448574176414, -0.031453374279633, -0.011448092005020, -0.088638964813854,  0.022741169835371,  0.000000000000000])
    orbit3 = jnp.array([0.0, -0.949478527562011,  0.000000000000000,  0.024541055790300,  0.077085096380257, -0.022172640589486, -0.053975565569078])
    orbit4 = jnp.array([0.0, -0.983172316307095,  0.000000000000000, -0.023578661445583, -0.074062151424169, -0.022172640589486,  0.051858876723232])   
    orbit5 = jnp.array([0.0,  0.000000000000000,  2.500000000000000,  2.097749077943200,  0.039063544778898,  0.000000000000000,  0.000000000000000])   
    masses = jnp.array([1, 0.95, 0.98, 1.02, 1.03])*mass_source

    # masses = jnp.array([1, 0.0001, 0.0003, 0.0008, 0.0002])  # (2,)
    # M=masses[0]

    #### circular orbits in one plane (xy)
    r1=3.8
    r2=1
    r3=1.9
    r4 =2.7
    # orbit1 = jnp.array([0.0, 0., 0.0, 0.0, 0.0, 0., 0.0])
    # orbit2 = jnp.array([0.0, r1, 0.0, 0.0, 0.0,orbital_circ_velocity(M,r1), 0.0])
    # orbit3 = jnp.array([0.0, 0, r2, 0, orbital_circ_velocity(M,r2) ,-0.0, 0.0])
    # orbit4 = jnp.array([0.0, 0, -r3, 0.0, -orbital_circ_velocity(M,r3),-0.0, 0.0])
    # orbit5 = jnp.array([0.0, -r4, 0, 0, 0.0,-orbital_circ_velocity(M,r4), 0.0])

    orbits = jnp.stack([orbit1, orbit2, orbit3, orbit4, orbit5])  
    h = 0.001
    T = 500

    N=20
    box_size=4.0
    low_mass = 0.1*mass_source
    high_mass =10*low_mass
    # orbits, masses = virialized_sphere_sampler(N, low_mass, high_mass, a=box_size/2, seed=0)
    # orbits, masses = plummer_sampler(N, key=jax.random.PRNGKey(1234), M1=0.5*mass_source, M2=2.0*mass_source, mass_dist="powerlaw", a=1.0)
    orbits, masses = plummer_sampler(N,key=jax.random.PRNGKey(1), M1=low_mass, M2=high_mass, a=box_size/2)
    COM_frame=True
    traj_com = integrate_nbody(orbits, masses, h, T, COM_frame)  
    plotname = "nbody_plummer"+str(N)
    from matplotlib import pyplot as plt
    def plot2d(masses, traj_com):
        for i in range(len(masses)):
            # print(i)
            plt.plot(traj_com[:, i, 1], traj_com[:, i, 2], markersize=.1, linewidth=1)
        plt.axis('equal')
        plt.xlim(-box_size,box_size)
        plt.ylim(-box_size,box_size)
        plt.title("n-body (COM frame), "+str(N)+" objects")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(plotname+".png", dpi=300)
        plt.close()

    plot = "both"    
    if plot == "2d":
        plot2d(masses, traj_com)  
    elif plot == "3d":
        plot_3d_trajectories(traj_com, plotname, box_size, elev=35, azim=45)
    elif plot == "both":
        plot2d(masses, traj_com)  
        plot_3d_trajectories(traj_com, plotname, box_size, elev=35, azim=45)

