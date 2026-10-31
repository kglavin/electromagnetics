# Module 6 recall sheet: induction, inductance, Maxwell's equations, waves

Blank sheet. No book, no notebook. Book pointer: chapters 14, 15, 17.

## Faraday's law (sections 14.1 to 14.4)

### Concepts
1. Emf as the line integral of the force per charge around a circuit. Why an emf is not a potential difference.
2. Motional emf: the force on the charges in a rod moving through a magnetic field, and the emf of a rod sliding on rails.
3. Magnetic flux through a loop, its units, and how the orientation of the surface fixes the sign of the emf.
4. Faraday's law in integral form. The two ways the flux can change: a moving or deforming loop, and a changing field.
5. Lenz's law: the direction of the induced current, argued from energy conservation.
6. The induced electric field: why it is not conservative, and why no potential can describe it.
7. Faraday's law in differential form, and how Stokes' theorem connects it to the integral form.
8. Eddy currents and magnetic braking: where the energy goes.

### Closed forms to remember
9. Emf of a rod of length L moving at speed v perpendicular to B.
10. Emf of a coil of N turns and area A rotating at angular frequency ω in a uniform field.
11. Flux through a coaxial loop of radius R from a dipole m on its axis at distance z.

### Algorithm
12. Flux through a loop from a sampled field: the quadrature over a disk, and why the samples must stay clear of the wires.
13. Emf from a flux time series by central differences, and the sign convention that goes with the loop's orientation.
14. Checking a flux-rule emf against the line integral of (v × B) · dl for a moving loop.

## Inductance and magnetic energy (sections 15.1 to 15.5)

### Concepts
15. Self-inductance defined from flux linkage. Why it depends only on geometry for linear materials.
16. Mutual inductance, and why M12 = M21.
17. The Neumann formula and where it comes from: the flux as a line integral of the vector potential.
18. The emf in a coil from its own changing current and from a neighbour's. The sign of each.
19. Transformers: the voltage ratio from the turns ratio, and what the ideal-transformer assumptions are.
20. Energy stored in an inductor, and the same energy written as an integral of the magnetic energy density.
21. The force between two current loops from the dependence of M on their separation.

### Closed forms to remember
22. Inductance of a long solenoid. Energy density of a magnetic field.
23. Mutual inductance of two coaxial loops far apart, from the dipole field.
24. Self-inductance of a thin ring: how it depends on the wire radius, and why a filament has infinite inductance.

### Algorithm
25. Neumann double sum over segments: cost, accuracy, and the condition that the two wires stay farther apart than a segment length.
26. Computing M two ways, by flux linkage and by the Neumann sum, and why they must agree.
27. M as a function of separation and tilt from a parameter sweep.

## Maxwell's equations (sections 17.1, 17.2)

### Concepts
28. The charging-capacitor paradox for Ampere's law, and how the displacement current resolves it.
29. Continuity of charge and why Ampere's law without the displacement term contradicts it.
30. The four Maxwell equations in integral and differential form, with what each says physically.
31. The Lorentz force as the link between the fields and the charges.
32. Which equations are constraints on the fields at one instant and which tell the fields how to evolve.

### Closed forms to remember
33. Displacement current density. The magnetic field between the plates of a charging circular capacitor.

## Electromagnetic waves

### Concepts
34. The wave equation for E and B in vacuum, derived from the curl equations.
35. The speed c from μ0 and ε0, and the numbers.
36. A plane wave: how E, B and the direction of travel are oriented, and the relation between the amplitudes.
37. Energy density of the field, the Poynting vector, and the intensity of a plane wave.
38. Waves in a dielectric: the speed, the refractive index, and the impedance.
39. Reflection and transmission at normal incidence on a dielectric step: amplitudes and energy.
40. A lossy medium: the attenuation of a wave and the skin depth of a good conductor.

### Closed forms to remember
41. c = 1/sqrt(μ0 ε0); η0 = sqrt(μ0/ε0) and its value; E/B = c.
42. Average intensity of a plane wave of amplitude E0.
43. Reflection and transmission coefficients for the electric field at normal incidence.

## The Yee method

### Concepts
44. The staggered grid: where E and H live in space and in time, and why staggering gives central differences for free.
45. The leapfrog: the order of the H and E updates in one time step, written out for 1D.
46. The Courant number and the stability limit in 1D and 2D. What an unstable run looks like.
47. Numerical dispersion: why short wavelengths on the grid travel at the wrong speed, and the rule of thumb for points per wavelength.
48. Boundaries: a perfect conductor, a first-order absorbing boundary, and a graded lossy layer. What each reflects.
49. Soft sources: why a source added to E radiates both ways in 1D.

### Algorithm
50. A 1D Yee update loop with a soft source and absorbing ends, from a blank editor.
51. Measuring the wave speed from probe arrival times, and why in 2D the delay between two probes is used.
52. Measuring reflection and transmission at a dielectric step from probe records.
53. The 2D TM update and the Poynting vector from the gridded fields.

### Self-check
Emf of a 100-turn coil of area 10 cm² in a field rising at 0.1 T/s. M of two coaxial 10 cm loops 5 cm apart, to the nearest 10 nH. Intensity of a plane wave with E0 = 100 V/m. Reflected field amplitude for a wave entering glass with ε_r = 4.
