# Module 1 recall sheet: fields, superposition, potential

Blank sheet. No book, no notebook. One section per sitting. Book pointers: chapters 1, 2, 3.

## Vector calculus (book chapter 1)


### Concepts
1. Dot product and cross product in components. Geometric meaning of each.
2. Scalar field versus vector field. One physical example of each.
3. Line integral, surface (flux) integral, volume integral: write the integrand and the differential element for each.
4. Gradient, divergence, curl in Cartesian components. Input type and output type of each.
5. Divergence theorem and Stokes' theorem as equations. Name the boundary in each.
6. The two identities that are always zero.

### Coordinate systems
7. Line element dl in cylindrical and spherical coordinates.
8. Unit vectors of cylindrical coordinates in terms of x-hat and y-hat.

### Algorithm
9. Build a 2D grid such that `P[i, j]` is at `(x[i], y[j])`. Which meshgrid argument matters?
10. Write the central difference for a first derivative. What is its error order?
11. From a single derivative routine, write divergence and curl_z of a 2D vector field.
12. Midpoint-rule line integral along a polyline: what are the two arrays you need?

### Self-check
Circulation of (-y, x) around the unit circle. Flux of (x, y) out of the unit square. State both numbers.

## Electric field (book chapter 2)


### Concepts
1. Coulomb's law as a vector equation. Direction of the unit vector. Value and units of 1/(4 pi eps0).
2. Definition of the electric field from the force on a test charge.
3. Superposition for N point charges as a sum.
4. Line, surface and volume charge density. Convert each to the charge of a small piece.
5. The integral for E from a volume charge density. Which position is primed?
6. Far-field behaviour of a finite distribution with net charge.

### Closed forms to remember
7. E on the axis of a uniformly charged ring.
8. E at perpendicular distance from the midpoint of a finite line of charge, and its infinite-line limit.
9. E of an infinite uniform plane.

### Algorithm
10. Discretise a continuous body into point charges: where do the pieces sit and what charge does each carry?
11. Write the broadcast that produces all N by M displacement vectors between observation points and sources.
12. What goes wrong if an observation point coincides with a source, and how is it handled?

### Self-check
For the ring, how fast does the on-axis error fall with N? Why is the in-plane error larger near the ring?

## Electric potential (book chapter 3)


### Concepts
1. What property of the electrostatic field makes it conservative? Give the differential and the integral statement.
2. Potential difference as a line integral of E, with sign.
3. V of a point charge and of a volume charge density, with V = 0 at infinity.
4. E = -grad V. Why the minus sign?
5. Relation between equipotential surfaces and field lines.
6. Potential energy of a test charge in a potential.

### Closed forms to remember
7. V on the axis of a uniformly charged ring, and its derivative as a check against the field from chapter 2.
8. V of a dipole at large distance, and how it scales with r compared to a monopole.

### Algorithm
9. Why computing V on a grid and differentiating is cheaper than computing E directly.
10. How to verify a gridded -grad V: two independent checks.
11. Expected error order of the central difference, and how to measure it by grid refinement.

### Self-check
Integrate E along three different paths between two fixed points. What should the three numbers equal?

