"""Independent micro-check of certificate identities (NumPy/SciPy only, not MFEM).

Problem: -Laplace(u) = f on the unit square, u = 0 on the boundary,
u = sin(pi x) sin(pi y), f = 2 pi^2 u. Uniform n x n grid, each square split
along the rising diagonal (the repository's Cartesian triangulation).

Computes, per n:
  e        energy error of the exact P1 Galerkin solution (uses the exact u; evaluation only)
  U_loc    ||q_loc - grad u_h|| for a vertex-patch RT0 equilibrated flux (P0 load projection)
  eta      guaranteed bound sqrt(sum_K (||q-grad u_h||_K + h_K/pi ||f-P0 f||_K)^2)
  U_opt    best RT0 certificate for this u_h (global stream-function solve)
  d_Q      ||q_opt - grad u||: RT0 dual discretization floor
  L_bub    lower bound from the Riesz representer of the residual in the edge-bubble space
  identity residual of U^2 = e^2 + ||q - grad u||^2 - 2 (f - P0 f, u - u_h)
  GS sweeps of local divergence-free (curl-of-hat) corrections needed to close 90% / 99%
  of the gap U_loc^2 - U_opt^2 (Gauss-Seidel on the Neumann P1 Laplacian).
"""
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

PI = np.pi
u_ex = lambda x, y: np.sin(PI * x) * np.sin(PI * y)
gu_ex = lambda x, y: np.stack([PI * np.cos(PI * x) * np.sin(PI * y), PI * np.sin(PI * x) * np.cos(PI * y)], -1)
f_ex = lambda x, y: 2 * PI**2 * np.sin(PI * x) * np.sin(PI * y)


def duffy_rule(m):
    """Collapsed Gauss rule on the reference triangle (0,0),(1,0),(0,1); exact to degree 2m-2."""
    g, w = np.polynomial.legendre.leggauss(m)
    g = (g + 1) / 2; w = w / 2
    s, t = np.meshgrid(g, g, indexing="ij"); ws, wt = np.meshgrid(w, w, indexing="ij")
    xi = s * (1 - t); eta = t; wq = ws * wt * (1 - t)
    return np.stack([xi.ravel(), eta.ravel()], 1), wq.ravel()


def mesh(n):
    xs = np.linspace(0, 1, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    V = np.stack([X.ravel(), Y.ravel()], 1)
    idx = lambda i, j: i * (n + 1) + j
    T = []
    for i in range(n):
        for j in range(n):
            sw, se, ne, nw = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            T.append([sw, se, ne]); T.append([sw, ne, nw])
    return V, np.array(T)


def run(n, gs_max=20000):
    V, T = mesh(n); NV, NT = len(V), len(T)
    P = V[T]                                   # (NT,3,2)
    J = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]], 2)   # columns
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    area = det / 2
    assert np.all(det > 0)
    Jinv = np.linalg.inv(J)
    gref = np.array([[-1., -1.], [1., 0.], [0., 1.]])
    G = np.einsum("kd,tde->tke", gref, Jinv)   # grad of barycentric lambda_k (NT,3,2)

    qp, qw = duffy_rule(8)
    lam = np.stack([1 - qp[:, 0] - qp[:, 1], qp[:, 0], qp[:, 1]], 1)    # (Q,3)
    X = np.einsum("qk,tkd->tqd", lam, P)                                   # (NT,Q,2)
    W = qw[None, :] * det[:, None]
    fq = f_ex(X[..., 0], X[..., 1])

    # ---- P1 Galerkin, exact solve
    Ke = area[:, None, None] * np.einsum("tid,tjd->tij", G, G)
    rows = np.repeat(T, 3, 1).ravel(); cols = np.tile(T, (1, 3)).ravel()
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), (NV, NV)).tocsr()
    Fe = np.einsum("tq,tq,qk->tk", W, fq, lam)
    F = np.bincount(T.ravel(), Fe.ravel(), NV)
    bnd = (np.abs(V[:, 0]) < 1e-14) | (np.abs(V[:, 0] - 1) < 1e-14) | (np.abs(V[:, 1]) < 1e-14) | (np.abs(V[:, 1] - 1) < 1e-14)
    free = ~bnd
    uh = np.zeros(NV); uh[free] = spla.spsolve(K[free][:, free].tocsc(), F[free])
    guh = np.einsum("tk,tkd->td", uh[T], G)                              # const per element
    e2 = np.sum(W * np.sum((gu_ex(X[..., 0], X[..., 1]) - guh[:, None, :])**2, -1))
    uhq = np.einsum("qk,tk->tq", lam, uh[T])
    err_q = u_ex(X[..., 0], X[..., 1]) - uhq

    # ---- edges with a fixed global normal
    loc_edges = [(1, 2), (2, 0), (0, 1)]          # edge k opposite local vertex k
    ekey = {}; Everts = []
    TE = np.zeros((NT, 3), int); S = np.zeros((NT, 3))
    for t in range(NT):
        for k, (a, b) in enumerate(loc_edges):
            va, vb = T[t, a], T[t, b]
            key = (min(va, vb), max(va, vb))
            if key not in ekey:
                ekey[key] = len(Everts); Everts.append(key)
            TE[t, k] = ekey[key]
    Everts = np.array(Everts); NE = len(Everts)
    # global normal: rotate tangent (from min vertex to max vertex) clockwise: n = (t_y, -t_x)
    tE = V[Everts[:, 1]] - V[Everts[:, 0]]; lenE = np.linalg.norm(tE, axis=1)
    nE = np.stack([tE[:, 1], -tE[:, 0]], 1) / lenE[:, None]
    for t in range(NT):
        c = P[t].mean(0)
        for k in range(3):
            E = TE[t, k]; mid = V[Everts[E]].mean(0)
            S[t, k] = 1.0 if np.dot(mid - c, nE[E]) > 0 else -1.0   # +1 if n_E outward of t
    midE = V[Everts].mean(1)
    bndE = (np.abs(midE[:, 0]) < 1e-14) | (np.abs(midE[:, 0] - 1) < 1e-14) | (np.abs(midE[:, 1]) < 1e-14) | (np.abs(midE[:, 1] - 1) < 1e-14)

    # ---- vertex-patch RT0 equilibration (P0 projection of the hat-weighted residual)
    mids = np.stack([(P[:, 1] + P[:, 2]) / 2, (P[:, 2] + P[:, 0]) / 2, (P[:, 0] + P[:, 1]) / 2], 1)  # (NT,3,2)
    mw = area / 3.0                                     # midpoint rule weight (exact for quadratics)
    # basis values at midpoints: phi_k(x) = S_k (x - P_k) / (2|K|)
    phi_mid = S[:, None, :, None] * (mids[:, :, None, :] - P[:, None, :, :]) / (2 * area[:, None, None, None])  # (NT,m,k,2)
    Mloc = np.einsum("t,tmid,tmjd->tij", mw, phi_mid, phi_mid)            # local RT0 mass
    lam_mid = np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]])          # lambda at midpoints
    psi_f = np.einsum("tq,tq,qk->tk", W, fq, lam)                        # int_K psi_k f
    patches = [[] for _ in range(NV)]
    for t in range(NT):
        for k in range(3):
            patches[T[t, k]].append((t, k))
    flux = np.zeros(NE)
    for a in range(NV):
        els = patches[a]
        edges = sorted({TE[t, k] for t, _ in els for k in range(3)})
        free_e = [E for E in edges if (a in Everts[E]) or (bnd[a] and bndE[E])]
        # boundary-of-patch edges not containing a and interior to Omega are fixed to zero
        fe = {E: i for i, E in enumerate(free_e)}
        ne, nt = len(free_e), len(els)
        A = np.zeros((ne, ne)); rhs = np.zeros(ne); Bm = np.zeros((nt, ne)); g = np.zeros(nt)
        for r, (t, ka) in enumerate(els):
            # psi_a grad u_h at midpoints
            target = lam_mid[:, ka][:, None] * guh[t][None, :]            # (m,2)
            for i in range(3):
                Ei = TE[t, i]
                if Ei not in fe: continue
                rhs[fe[Ei]] += np.sum(mw[t] * np.sum(phi_mid[t, :, i] * target, -1))
                Bm[r, fe[Ei]] += S[t, i]
                for j in range(3):
                    Ej = TE[t, j]
                    if Ej in fe: A[fe[Ei], fe[Ej]] += Mloc[t, i, j]
            g[r] = area[t] * np.dot(G[t, ka], guh[t]) - psi_f[t, ka]      # |K| * P0(grad psi.grad u_h - psi f)
        interior = not bnd[a]
        if interior:
            Z = np.zeros((ne + nt + 1, ne + nt + 1))
            Z[:ne, :ne] = A; Z[:ne, ne:ne + nt] = Bm.T; Z[ne:ne + nt, :ne] = Bm
            Z[ne:ne + nt, -1] = 1; Z[-1, ne:ne + nt] = 1
            sol = np.linalg.solve(Z, np.concatenate([rhs, g, [0.]]))
        else:
            Z = np.zeros((ne + nt, ne + nt))
            Z[:ne, :ne] = A; Z[:ne, ne:] = Bm.T; Z[ne:, :ne] = Bm
            sol = np.linalg.solve(Z, np.concatenate([rhs, g]))
        for E, i in fe.items():
            flux[E] += sol[i]

    fl_t = flux[TE]                                                      # (NT,3)
    div_q = np.sum(S * fl_t, 1) / area
    P0f = np.sum(W * fq, 1) / area
    eq_res = np.max(np.abs(div_q + P0f)) / np.max(np.abs(P0f))

    def q_at(fl_t_, pts):  # pts (NT,Q,2)
        return np.einsum("tk,tkqd->tqd", fl_t_ * S / (2 * area[:, None]), pts[:, None, :, :] - P[:, :, None, :])

    qX = q_at(fl_t, X)
    r_el = np.sum(W * np.sum((qX - guh[:, None, :])**2, -1), 1)
    U_loc2 = np.sum(r_el)
    osc_el = (np.sqrt(2) / n / PI) * np.sqrt(np.sum(W * (fq - P0f[:, None])**2, 1))
    eta = np.sqrt(np.sum((np.sqrt(r_el) + osc_el)**2))
    dq_loc2 = np.sum(W * np.sum((qX - gu_ex(X[..., 0], X[..., 1]))**2, -1))
    cross = np.sum(W * (fq - P0f[:, None]) * err_q)
    ident_res = abs(U_loc2 - (e2 + dq_loc2 - 2 * cross)) / U_loc2

    # ---- dual tightening: curl of P1 hats; Gram matrix = Neumann P1 stiffness K
    # curl phi = (phi_y, -phi_x), constant per element
    curl = np.stack([G[..., 1], -G[..., 0]], -1)                        # (NT,3,2)
    r0_int = np.einsum("tqd->td", W[..., None] * (qX - guh[:, None, :]))  # int_K r0
    be = np.einsum("tkd,td->tk", curl, r0_int)
    b = np.bincount(T.ravel(), be.ravel(), NV)
    KN = K  # Neumann stiffness = full K (all vertices)
    # optimal: min ||r0 + curl psi||^2 -> KN psi = -b (singular, consistent)
    pin = 0
    keep = np.arange(NV) != pin
    psi_opt = np.zeros(NV); psi_opt[keep] = spla.spsolve(KN[keep][:, keep].tocsc(), -b[keep])
    U_opt2 = U_loc2 + 2 * b @ psi_opt + psi_opt @ (KN @ psi_opt)
    curl_psi = np.einsum("tk,tkd->td", psi_opt[T], curl)
    q_opt_X = qX + curl_psi[:, None, :]
    dQ2 = np.sum(W * np.sum((q_opt_X - gu_ex(X[..., 0], X[..., 1]))**2, -1))
    # equilibration of q_opt unchanged (divergence-free correction); check directly via fluxes:
    # flux of curl psi through edge = psi(max vertex) - psi(min vertex)
    dflux = psi_opt[Everts[:, 1]] - psi_opt[Everts[:, 0]]
    div_opt = np.sum(S * (flux + dflux)[TE], 1) / area
    eq_res_opt = np.max(np.abs(div_opt + P0f)) / np.max(np.abs(P0f))
    U_opt2_direct = np.sum(W * np.sum((q_opt_X - guh[:, None, :])**2, -1))

    # Gauss-Seidel sweeps (local vertex corrections with exact one-dimensional gains)
    Lw = sp.tril(KN, 0).tocsr(); Up = sp.triu(KN, 1).tocsr()
    psi = np.zeros(NV); gap0 = U_loc2 - U_opt2
    sweeps = {0.9: None, 0.99: None}
    for s_ in range(1, gs_max + 1):
        psi = spla.spsolve_triangular(Lw, -b - Up @ psi, lower=True)
        U2 = U_loc2 + 2 * b @ psi + psi @ (KN @ psi)
        closed = 1 - (U2 - U_opt2) / gap0
        for k in sweeps:
            if sweeps[k] is None and closed >= k: sweeps[k] = s_
        if all(v is not None for v in sweeps.values()): break

    # smooth (low-frequency) dual slack: start GS from psi_opt + psi_s, psi_s = 0.1*cos(pi x)cos(pi y)
    psi_s = 0.1 * np.cos(PI * V[:, 0]) * np.cos(PI * V[:, 1])
    psi = psi_opt + psi_s; gap_s0 = psi_s @ (KN @ psi_s); smooth = {0.9: None}
    for s_ in range(1, gs_max + 1):
        psi = spla.spsolve_triangular(Lw, -b - Up @ psi, lower=True)
        d = psi - psi_opt; d = d - d.mean()
        gap = d @ (KN @ d)
        if smooth[0.9] is None and gap <= 0.1 * gap_s0: smooth[0.9] = s_; break
    # ---- lower bound: edge-bubble witness (Riesz representer in bubble space, exact solve)
    # bubble b_E = 4 lam_i lam_j on each element containing E (interior edges only)
    intE = ~bndE
    lamq = lam[None].repeat(NT, 0)                                       # (NT,Q,3)
    Rb = np.zeros(NE); rowsB, colsB, valsB = [], [], []
    gradb = np.zeros((NT, 3, qw.size, 2)); valb = np.zeros((NT, 3, qw.size))
    for k, (i, j) in enumerate(loc_edges):
        valb[:, k] = 4 * lamq[..., i] * lamq[..., j]
        gradb[:, k] = 4 * (lamq[..., i][..., None] * G[:, None, j] + lamq[..., j][..., None] * G[:, None, i])
    Rloc = np.einsum("tq,tq,tkq->tk", W, fq, valb) - np.einsum("tkqd,td,tq->tk", gradb, guh, W)
    Bloc = np.einsum("tq,tkqd,tlqd->tkl", W, gradb, gradb)
    Rb = np.bincount(TE.ravel(), Rloc.ravel(), NE)
    rB = np.repeat(TE, 3, 1).ravel(); cB = np.tile(TE, (1, 3)).ravel()
    Bm_ = sp.coo_matrix((Bloc.ravel(), (rB, cB)), (NE, NE)).tocsr()
    ii = np.where(intE)[0]
    c = spla.spsolve(Bm_[ii][:, ii].tocsc(), Rb[ii])
    L_bub = np.sqrt(Rb[ii] @ c)
    # Jacobi (diagonal) witness: one cheap step
    cj = Rb[ii] / Bm_[ii][:, ii].diagonal()
    L_jac = abs(Rb[ii] @ cj) / np.sqrt(cj @ (Bm_[ii][:, ii] @ cj))
    # shift identity check: v' = u_h + beta*w, true e'^2 vs e^2 - L^2 (exact load) and data defect
    cfull = np.zeros(NE); cfull[ii] = c
    wq = np.einsum("tk,tkq->tq", cfull[TE], valb); gwq = np.einsum("tk,tkqd->tqd", cfull[TE], gradb)
    e2_new = np.sum(W * np.sum((gu_ex(X[..., 0], X[..., 1]) - guh[:, None, :] - gwq)**2, -1))
    shift_res = abs(e2_new - (e2 - L_bub**2)) / e2
    data_term = 2 * np.sum(W * (fq - P0f[:, None]) * wq)                 # U-gain defect with P0 load
    U_new2 = np.sum(W * np.sum((qX - guh[:, None, :] - gwq)**2, -1))
    Ushift_res = abs((U_loc2 - U_new2) - (L_bub**2 - data_term)) / U_loc2

    e = np.sqrt(e2)
    return dict(n=n, dofs=int(free.sum()), e=e, U_loc=np.sqrt(U_loc2), eta=eta,
                eff_loc=np.sqrt(U_loc2) / e, eff_eta=eta / e, U_opt=np.sqrt(U_opt2),
                U_opt_direct=np.sqrt(U_opt2_direct), eff_opt=np.sqrt(U_opt2) / e,
                dQ_over_e=np.sqrt(dQ2) / e, osc=np.sqrt(np.sum(osc_el**2)),
                L_bub=L_bub, effL=L_bub / e, L_jac=L_jac, effLjac=L_jac / e,
                eq_res=eq_res, eq_res_opt=eq_res_opt, ident_res=ident_res,
                shift_res=shift_res, Ushift_res=Ushift_res, gs90=sweeps[0.9], gs99=sweeps[0.99], gs90_smooth=smooth[0.9])


if __name__ == "__main__":
    import json
    ns = [int(a) for a in sys.argv[1:]] or [8, 16, 32, 64]
    out = [run(n) for n in ns]
    for r in out:
        print(json.dumps({k: (float(f"{v:.6g}") if isinstance(v, float) else v) for k, v in r.items()}))
