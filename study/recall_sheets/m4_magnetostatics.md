# Module 4 recall sheet: magnetic field, Ampere's law, magnetization

Blank sheet. No book, no notebook. Book pointer: chapters 11, 12, 13.

## The magnetic field and magnetic forces (sections 11.1, 11.2)

### Concepts
1. The Lorentz force on a moving charge. Why a magnetic field does no work on a charge.
2. Units of B. The sizes of the Earth's field, a fridge magnet, an MRI magnet.
3. Motion of a charge in a uniform B: shape of the orbit, which quantities are constant, why the period does not depend on speed.
4. Crossed E and B: the drift velocity, its direction, and why it is the same for both signs of charge.
5. Force on a current element I dl in a field. Force per length between two parallel wires, and the sign for parallel versus antiparallel currents.
6. Torque on a current loop. Definition of the magnetic dipole moment and its direction by the right-hand rule.

### Closed forms to remember
7. Cyclotron frequency and Larmor radius in terms of q, m, v_perp, B.
8. E x B drift velocity.
9. Torque and potential energy of a dipole in a uniform field.

### Algorithm
10. The Boris pusher: the three sub-steps and why the rotation step preserves speed exactly.
11. Why RK4 drifts in energy over many periods and Boris does not. Which one is still second order.
12. How a field given on a grid or as a callable enters the pusher.

## Biot-Savart and the vector potential (sections 11.1, 11.3)

### Concepts
13. The Biot-Savart law for a current element. Which vector appears in the cross product and which power of distance in the denominator.
14. Why div B = 0 follows from Biot-Savart, and what it means for field lines.
15. The vector potential: definition, the gauge freedom, and the analogue of Poisson's equation it satisfies.

### Closed forms to remember
16. B of an infinite straight wire. B on the axis of a circular loop. The far field of a loop as a dipole with m = I A.
17. Helmholtz pair: separation, field at the centre, and which derivatives vanish there.
18. B on the axis of a finite solenoid with the two angles, and the infinite-solenoid limit.

### Algorithm
19. A wire as a polyline: midpoints and dl vectors, closed versus open, and the one-line vectorised sum.
20. Why the sum is evaluated in chunks, and what happens at a point on the wire itself.
21. The closed-form checks for a loop, a wire and a solenoid, and the resolution needed near a wire.

## Flux and Ampere's law (sections 12.1 to 12.4)

### Concepts
22. Magnetic flux through a surface. Why the flux through a closed surface is zero, in both forms.
23. Ampere's law in integral and differential form. Which current is enclosed and what sign convention fixes it.
24. The symmetry argument for a long straight wire, a long solenoid and a toroid: what the Amperian loop looks like and why B is constant on it.
25. Why Ampere's law alone does not give B for a finite loop or a short solenoid.
26. The comparison table: Gauss and Ampere, E and B, scalar and vector potential, sources and symmetries.
27. Stokes' theorem as the link between the two forms (from the module 1 primer).

### Closed forms to remember
28. B inside and outside a long solenoid, inside a toroid, and inside a wire of radius a carrying uniform current.

### Algorithm
29. Circulation of a gridded or callable B around a loop with a midpoint sum, and the resolution needed close to a wire.
30. Checking curl B = mu_0 J on a grid where J is a smooth current density.

## Magnetization (sections 13.1 to 13.5)

### Concepts
31. Magnetization M: definition, units, and the two microscopic origins of magnetic moment.
32. Bound volume current J_b = curl M and bound surface current K_b = M x n: the loop-counting argument for both.
33. Why a uniformly magnetized bar is equivalent to a solenoid of surface current.
34. The auxiliary field H: definition, Ampere's law in terms of free current, and why H is not simply B / mu_0.
35. Linear materials: chi_m, mu_r, the sign of chi_m for diamagnets and paramagnets, and why ferromagnets have no single chi_m.
36. Hysteresis: what the loop records, remanence and coercivity in one sentence each.
37. Boundary conditions at an interface: which component of B and which of H is continuous, and what a free surface current changes.
38. A magnetic circuit: the analogue of Ohm's law, reluctance of a path and of an air gap.

### Closed forms to remember
39. Uniformly magnetized sphere: interior B and H, and the exterior dipole moment.
40. Toroid with a core of permeability mu: B inside, and the effect of a small air gap.

### Algorithm
41. J_b from a gridded M: which module-1 routine, and the test field with a known curl.
42. Recovering K_b from the discrete curl at a step in M, in the same way sigma_b came from the divergence of P.
43. The field of a magnetized object as the Biot-Savart field of its bound currents.

### Self-check
Larmor radius of a 1 keV proton in 0.1 T. B at the centre of a Helmholtz pair with R = 10 cm and I = 1 A. K_b on the side of a bar magnet with M = 1e6 A/m. B inside a long solenoid with 1000 turns per metre and 2 A, empty and with mu_r = 500.
