# Module 2 recall sheet: Gauss, Laplace, conductors, capacitance

Blank sheet. No book, no notebook. One section per sitting. Book pointers: chapters 4, 6, 7, 8.

## Gauss' law (book chapter 4)

### Concepts
1. Electric flux through an open surface and through a closed surface. Sign convention.
2. Gauss' law in integral form. What counts as enclosed charge?
3. From integral to differential form via the divergence theorem, in three lines.
4. The three symmetric geometries and the Gaussian surface for each.

### Closed forms to remember
5. Field of a uniformly charged sphere inside and outside. Field of an infinite line and an infinite plane.

### Algorithm
6. Flux through an axis-aligned box on a grid: which field component on which face, and with what sign?
7. Two checks that the numerical flux is right.

## Laplace and Poisson (book chapter 6)

### Concepts
8. Derive Poisson's equation. Dirichlet versus Neumann conditions, with one physical example each.
9. The uniqueness theorem and what it buys you numerically.

### Algorithm
10. The five-point stencil with the source term. Units of h^2 rho / eps0.
11. Jacobi versus Gauss-Seidel versus SOR: what differs in one sentence each, and how sweep counts scale with grid size.
12. How fixed nodes enter a relaxation solver and how they enter a sparse matrix solve.

## Conductors (book chapter 7)

### Concepts
13. Four properties of an ideal conductor in equilibrium with one-line reasons.
14. sigma = eps0 E_n: derive it with a pillbox.
15. Mirror charge for a grounded plane and for a grounded sphere. Which two conditions make the image solution the solution?

### Algorithm
16. Three ways to get induced charge from a solved potential grid.

## Capacitance (book chapter 8)

### Concepts
17. Definition of C. Parallel plate, coaxial, and spherical capacitance.
18. Energy of a capacitor in three forms. Field energy density.
19. Series and parallel rules.

### Algorithm
20. Capacitance from field energy: the four steps.
21. Why a floating conductor sits at the potential that minimises field energy.

### Self-check
Flux through any box around q. Sweep scaling of SOR. Total induced charge on a grounded plane from a line charge lambda. C of coax with b = 4a.
