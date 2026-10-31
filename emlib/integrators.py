"""Charged-particle motion in electric and magnetic fields: Boris pusher and RK4."""

import numpy as np


def _field(F):
    """Accept a constant vector or a callable (N, 3) -> (N, 3) and return a callable."""
    if callable(F):
        return F
    const = np.asarray(F, dtype=float)
    return lambda x: np.broadcast_to(const, x.shape).copy()


def lorentz_acceleration(x, v, q, m, E, B):
    """q/m (E + v x B) at positions x with velocities v, both (N, 3)."""
    E, B = _field(E), _field(B)
    return (q / m) * (E(x) + np.cross(v, B(x)))


def boris(x0, v0, q, m, E, B, dt, n_steps):
    """Boris pusher. Returns positions and velocities, each (n_steps + 1, 3).

    The velocity is advanced by a half electric kick, a rotation about B that preserves speed
    exactly, and another half kick; the position then moves with the new velocity. The rotation
    is what makes the method conserve energy in a pure magnetic field for any step size.
    As in any leapfrog scheme, v[k] is best read as the velocity half a step before x[k];
    the offset is first order in dt and shows up as a small shift of a circular orbit's centre.
    """
    E, B = _field(E), _field(B)
    x = np.zeros((n_steps + 1, 3))
    v = np.zeros((n_steps + 1, 3))
    x[0], v[0] = np.asarray(x0, dtype=float), np.asarray(v0, dtype=float)
    qm = q / m
    for k in range(n_steps):
        pos = x[k][None, :]
        e, b = E(pos)[0], B(pos)[0]
        v_minus = v[k] + 0.5 * qm * e * dt
        t = 0.5 * qm * b * dt
        s = 2.0 * t / (1.0 + t @ t)
        v_prime = v_minus + np.cross(v_minus, t)
        v_plus = v_minus + np.cross(v_prime, s)
        v[k + 1] = v_plus + 0.5 * qm * e * dt
        x[k + 1] = x[k] + v[k + 1] * dt
    return x, v


def rk4(x0, v0, q, m, E, B, dt, n_steps):
    """Classical fourth-order Runge-Kutta on (x, v) with the Lorentz force. Same return shape as boris."""
    E, B = _field(E), _field(B)
    x = np.zeros((n_steps + 1, 3))
    v = np.zeros((n_steps + 1, 3))
    x[0], v[0] = np.asarray(x0, dtype=float), np.asarray(v0, dtype=float)

    def acc(pos, vel):
        return lorentz_acceleration(pos[None, :], vel[None, :], q, m, E, B)[0]

    for k in range(n_steps):
        xk, vk = x[k], v[k]
        k1x, k1v = vk, acc(xk, vk)
        k2x, k2v = vk + 0.5 * dt * k1v, acc(xk + 0.5 * dt * k1x, vk + 0.5 * dt * k1v)
        k3x, k3v = vk + 0.5 * dt * k2v, acc(xk + 0.5 * dt * k2x, vk + 0.5 * dt * k2v)
        k4x, k4v = vk + dt * k3v, acc(xk + dt * k3x, vk + dt * k3v)
        x[k + 1] = xk + dt / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
        v[k + 1] = vk + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return x, v


def rk4_ode(f, y0, t):
    """Classical RK4 for dy/dt = f(t, y) on the time grid t. Returns y with shape (len(t),) + shape(y0)."""
    t = np.asarray(t, dtype=float)
    y0 = np.asarray(y0)
    y = np.zeros((len(t),) + y0.shape, dtype=np.result_type(y0.dtype, float))
    y[0] = y0
    for k in range(len(t) - 1):
        dt, tk, yk = t[k + 1] - t[k], t[k], y[k]
        k1 = f(tk, yk)
        k2 = f(tk + 0.5 * dt, yk + 0.5 * dt * k1)
        k3 = f(tk + 0.5 * dt, yk + 0.5 * dt * k2)
        k4 = f(tk + dt, yk + dt * k3)
        y[k + 1] = yk + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def cyclotron_frequency(q, B, m):
    """Angular frequency |q| B / m of circular motion in a uniform field of magnitude B."""
    return abs(q) * B / m


def larmor_radius(m, v_perp, q, B):
    """Radius m v_perp / (|q| B) of the circle traced perpendicular to a uniform field."""
    return m * v_perp / (abs(q) * B)
