# Certified floor attribution for the P1 and RT0 pair

## Problem and proof boundary

On the unit square let $u\in H^1_0$ solve $(a\nabla u,\nabla w)=(f,w)$ for every $w\in H^1_0$. The constant $a>0$, tolerance $\epsilon>0$, mesh coordinates, cell loads and action budget are frozen. The target is $\|u-v\|_a$ with $\|v\|_a^2=\int a|\nabla v|^2$. The primary contract uses a rational P0 load, a conforming straight triangular mesh, a continuous P1 field with zero trace and an RT0 flux. The checker receives a problem digest retained before candidate generation. Coordinates, fields and moments denote exact rationals, including the exact binary values of finite floats. Sampling or measurement uncertainty would require an additional input enclosure.

Each triangle has positive orientation and the checker requires exact unit area in total. Interior edges occur twice with opposite orientations. Remaining edges tile each of the four sides exactly once. All vertices are distinct, used, and lie in the square. The oriented triangle boundaries therefore sum to the positively oriented square. Away from the edges, the winding number of this chain counts the covering triangles, each positively. Its value is one inside and zero outside. This excludes overlap and holes. Complete edge pairing then excludes a hanging edge or a T-junction. Edge keys and global vertex values give a single-valued trace.

For an edge $i<j$, rotate its tangent clockwise to define the normal. Store the integral of its normal flux once per edge. If $s_{K,e}$ is the local outward sign, then
$$q|_K(x)=\sum_{e\subset\partial K}s_{K,e}F_e\frac{x-p_{K,e}}{2|K|},\qquad
 |K|\operatorname{div}q=\sum_e s_{K,e}F_e,$$
where $p_{K,e}$ is the opposite vertex. An RT0 normal trace is constant on each edge, so sharing a moment proves interior normal continuity. The sign convention requires $\operatorname{div}q=-f$. Dirichlet data allow free boundary flux.

## Independent energy certificate

Write $p=a\nabla u$ and $G=\|q-a\nabla v\|_{a^{-1}}^2$. Exact equilibrium makes $q-p$ orthogonal to all gradients in $H^1_0$. Expanding the square proves the hypercircle identity
$$G=\|u-v\|_a^2+\|q-p\|_{a^{-1}}^2.$$
The checker accepts the current field only when its independently validated squared upper bound is at most the frozen $\epsilon^2$. Failure of a mathematical input check yields no certified acceptance. A certified upper bound above tolerance still contains information about that field.

Under imperfect equilibrium let $\delta=f+\operatorname{div}q$. Testing the error equation with $u-v$ gives
$$\|u-v\|_a\ \le\ \sqrt G + \|\delta\|_{L^2}/\sqrt{18a}.$$
Here the exact Dirichlet Poincaré constant of the unit square is $1/(\pi\sqrt2)$ and $\pi>3$. For an affine original $f$ the checker computes and checks the exact mean $f_h$ on each triangle. Since $f_h+\operatorname{div}q$ is cellwise constant, its squared norm and $\|f-f_h\|^2$ add exactly to $\|\delta\|^2$. Both are integrated with the degree-two midpoint rule, as is $G$. For $c=\|\delta\|^2/(18a)$ the checker compares $G+c+2\operatorname{sqrtUp}(Gc)$ with $\epsilon^2$. Rational square roots are exact when representable. Otherwise the returned multiple of $2^{-96}$ is rounded upwards using integer square roots. Zero gets an exact zero correction.

For an exactly represented P2 function $z$ with zero trace, define $R(z)=(f,z)-(a\nabla v,\nabla z)$. Cauchy–Schwarz gives $\|u-v\|_a^2\ge R(z)^2/\|z\|_a^2$. A zero witness contributes zero. Shared vertex and midpoint values enforce conformity and the degree-three triangle rule integrates the affine-load pairing exactly. The P2 witness is diagnostic. Adding it to a P1 field would require declaring and checking a different primal space.

## Floors of the current pair

Let $v_*$ be the Galerkin solution in $V_h$. Let $q_*$ minimize $\|q-p\|_{a^{-1}}$ among equilibrated RT0 fluxes $Q_h(f)$. Primal Galerkin orthogonality splits the primal error. Dual minimization over an affine space splits the flux error. Consequently
$$G=E_{\rm disc}^2+E_{\rm alg}^2+D_{\rm disc}^2+D_{\rm alg}^2,$$
where the four terms are the squared norms of $u-v_*,v_*-v,p-q_*,q_*-q$ in their respective metrics. The pair floor $\Phi=E_{\rm disc}^2+D_{\rm disc}^2$ depends only on the spaces and the frozen problem. Minimizing the gap in the dual nullspace requires no primal optimum. Every divergence-free RT0 perturbation is a rotated gradient of a continuous P1 stream function on this simply connected mesh. In edge coordinates its moments are $\psi_j-\psi_i$. Conversely, zero cell divergence gives path-independent differences on the triangulated disk. Streams differing by a constant have the same curl. Pinning one vertex yields an SPD dual problem with energy coefficient $1/a$.

For each reduced SPD system with residual $r$ and matrix $K$, algebraic error squared equals $r^TK^{-1}r$. A line minimization in direction $D^{-1}r$ gives an attained lower bound $g$ and a remaining residual $r'$. For a node at edge distance $d_i$ from a zero anchor, summing edge differences gives $|z_i|\le d_i\sqrt M\,\|z\|_K$. One may take $M=\max_{K,e}|e|^2/(a_K |K|)$ using the energy coefficient of the system. Thus $g+M(\sum_i d_i|r'_i|)^2$ is an upper bound. The total exact-equilibrium gap also caps each component. The optional small-system control evaluates the residual inverse energy by rational elimination. For the dual equations the checker verifies that the complete right-hand side annihilates constants before deleting the pinned coordinate. Both energy and residual bounds use the same pinned coordinates.

With squared component intervals $[A_-,A_+]$ and $[B_-,B_+]$, on the same immutable primal–dual state,
$$\max(0,G_- - A_+ - B_+)\le\Phi\le G_+ - A_- - B_-.$$
The rational exact-equilibrium path has $G_-=G_+=G$. A witness gives $E_{\rm disc}^2\ge\max(0,R(z)^2/\|z\|_a^2-A_+)$. The checks are applied in this order.

| Condition | Conclusion about the checked state or space |
|---|---|
| Certified upper bound $\le\epsilon^2$ | Accept the current primal field. |
| Witness primal floor lower bound $>\epsilon^2$ | Every field in this primal space misses the target. |
| Pair floor lower bound $>\epsilon^2$ | No member of this pair attains the gap target. |
| Pair floor upper bound $\le\epsilon^2$ | The optimal pair can attain the target. The current field still needs its own inspection. |
| Interval straddles the target | The evidence leaves the pair decision undetermined. |

## Projected and original problems

For $u_h^{\rm cont}$ solving the continuous PDE with $f_h$, exact equilibrium against $f_h$ gives the four-term decomposition **for that projected problem**, including Galerkin and dual optima for that load. For the original solution $u$ and a flux equilibrated against $f_h$, direct expansion instead gives
$$G=\|u-v\|_a^2+\|q-a\nabla u\|_{a^{-1}}^2-2(f-f_h,u-v).$$
The data term has no fixed sign. The corrected upper bound is suitable for original-field acceptance. Only a separately frozen projected problem may use the uncorrected gap to attribute its floors when the original oscillation is nonzero. The same numerical field and flux can therefore give different verdicts under the two problem digests. A nonzero equilibrium defect likewise disables floor inference. A general non-polynomial source currently lacks the representation and validated integrations required to certify its original PDE.

## Independent continuous reference for evaluation

For constant $f$ the Dirichlet sine series bounds $S=\|u\|_a^2$ without inserting an over-resolved FE solution. For odd positive $m,n$, sum $64f^2/(a\pi^6m^2n^2(m^2+n^2))$. After a square cutoff $N$, the remaining dimensionless sum is bounded above by $4/(3N^3)$ using the union of the two tails and integral comparisons. Alternating arctangent bounds in Machin's formula enclose $\pi$. The two discretization terms are $S-\|v_*\|_a^2$ and $\|q_*\|_{a^{-1}}^2-S$. Their sum is exactly rational even though the separate continuous terms are enclosed. This harness is isolated from solving, certification and decisions.
