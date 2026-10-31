# Module 5 recall sheet: current, circuits, disordered media

Blank sheet. No book, no notebook. Book pointer: chapters 9, 10, 16.

## Current and resistance (sections 9.1 to 9.5)

### Concepts
1. Current and current density: definitions, units, and the relation I = ∫ J · dA.
2. Drift velocity: why it is tiny compared with the thermal speed of the electrons, and why the current still starts almost at once when a switch closes.
3. Ohm's law in microscopic form, J = σ E. The collision picture that gives σ in terms of n, e, m and the time between collisions.
4. Resistance of a uniform conductor from J = σ E, and how R scales with length and cross-section.
5. Power delivered to a resistor in three forms, and the local power density J · E.
6. Conservation of charge in integral and differential form. What div J = 0 means for a steady current.
7. Why the interior of an ohmic conductor carrying a steady current has no net charge, and where the charge that shapes the field sits instead.
8. Boundary conditions at an interface between two conductors: which component of J is continuous, and what happens to E.

### Closed forms to remember
9. Drift speed for a given current density and carrier density. The order of magnitude for copper at 1 A/mm².
10. R of a straight wire. R between two coaxial cylinders, and between two concentric spheres, filled with a conductor of conductivity σ.
11. Charge relaxation time in a conductor, ε0/σ, and its value for copper.

### Algorithm
12. A conducting sheet as a resistor network: what the conductance of one face is, and why the harmonic mean of the two node conductivities is used.
13. How the current is measured on a grid, and the check that the same total current crosses every column.
14. How a hole or an insulating edge enters the network.

## DC circuits and RC circuits (sections 10.1 to 10.3)

### Concepts
15. EMF versus potential difference. A real battery as an ideal source in series with an internal resistance.
16. Kirchhoff's current law and voltage law, and which field equation each one comes from.
17. Series and parallel combinations of resistors: what is shared, what adds.
18. Nodal analysis by hand: unknowns, one equation per node, and why the result is a symmetric linear system.
19. Thevenin and Norton equivalents: how to find them from two measurements.
20. The Wheatstone bridge: the balance condition and why it is independent of the source voltage.
21. Charging and discharging a capacitor through a resistor: the differential equation and its solution.
22. Energy bookkeeping during charging: what the source supplies, what the capacitor stores, what the resistor dissipates.
23. Circuits as models: which physical assumptions turn a piece of wire into a node.

### Closed forms to remember
24. V_C(t) and I(t) for charging and discharging. The values at t = RC and t = 5 RC.
25. Infinite ladder of identical resistors: the input resistance.

### Algorithm
26. Modified nodal analysis: how a resistor, a current source and a voltage source are stamped into the matrix, and why voltage sources need extra unknowns.
27. Choosing a ground node, and what goes wrong without one.
28. An RC transient as an ODE integrated with RK4, checked against the closed form.

## AC circuits (sections 16.1 to 16.4)

### Concepts
29. The three passive components: the voltage-current relation of each, and the energy each stores or dissipates.
30. The inductor as a component: V = L dI/dt, why the current through it cannot jump, and where L comes from (chapters 14 and 15).
31. RL and LC circuits in the time domain. The series RLC equation and its underdamped, critical and overdamped solutions.
32. Sinusoidal steady state: phasors, what the complex amplitude records, and how a time derivative becomes multiplication by jω.
33. Impedance of R, L and C. Combining impedances in series and parallel.
34. Transfer function of an RC low-pass and high-pass filter. Cut-off frequency, the slope in decibels per decade on a Bode plot, and the phase.
35. Series RLC resonance: what cancels at ω0, the current at resonance, the quality factor and the bandwidth.
36. Average power and RMS values. Why a capacitor or inductor takes no average power.
37. The assumptions behind lumped circuit models: size compared with wavelength, quasi-static fields, ideal components. One example where each fails.

### Closed forms to remember
38. ω0, Q and the half-power bandwidth of a series RLC.
39. |H| and phase of an RC low-pass at ω = 1/RC.
40. The decay rate and the oscillation frequency of an underdamped series RLC.

### Algorithm
41. AC nodal analysis: the same solve with complex admittances, and how to read magnitude and phase off the result.
42. A Bode sweep: the frequency grid, and the check of one stage against its closed form.
43. Comparing a time-domain RK4 solution after the transient has died with the phasor amplitude.

## Disordered media and percolation (section 10.4)

### Concepts
44. A random resistor network as a model of a composite, a porous rock or a thin film. Bond versus site percolation.
45. Clusters, the spanning cluster, and the percolation threshold. Why p_c = 1/2 for bonds on the square lattice.
46. Backbone versus dangling ends: which bonds of the spanning cluster carry current, and why.
47. How the effective conductivity behaves below, at and above the threshold. The effective-medium estimate and where it fails.
48. Finite-size effects: why a finite lattice spans below p_c in some samples and above it in others.

### Closed forms to remember
49. Effective-medium conductivity of the bond lattice, 2(p − 1/2), and the normalisation that gives 1 for the full lattice.

### Algorithm
50. Building the lattice from random numbers, one per bond, so that bonds only appear as p rises.
51. Kirchhoff's law on the lattice: why isolated clusters must be removed or pinned before the sparse solve.
52. Detecting a spanning cluster with connected components, and finding the backbone from the solved currents.
53. Averaging σ_eff over samples, and how many samples are needed near the threshold.

### Self-check
Drift speed in a 1 mm² copper wire carrying 10 A. R between coaxial cylinders of radii 1 mm and 5 mm, length 1 m, filled with σ = 1 S/m. V_C after 2 RC when charging from 0 to 9 V. ω0 and Q for R = 10 Ω, L = 10 mH, C = 10 μF. σ_eff of the bond lattice at p = 0.8 by the effective-medium estimate.
