import itertools, numpy as np
def T_sym(C, eta):
    T = np.full((C, C), eta/(C-1)); np.fill_diagonal(T, 1-eta); return T
def T_pair(C, eta):
    T = np.eye(C)*(1-eta)
    for i in range(C): T[i,(i+1)%C] += eta
    return T
def U_part(z, parts, C):
    u = 0.0
    for B in parts:
        n = len(B)
        if n == 1: u += (1-1/C); continue          # singleton: no purity evidence
        cnt = np.bincount([z[i] for i in B], minlength=C)
        u += n*(1-(cnt*(cnt-1)).sum()/(n*(n-1)))
    return u
def G_part(z, parts, C):
    g = 0.0
    for B in parts:
        n = len(B); cnt = np.bincount([z[i] for i in B], minlength=C)
        g += n*(1-(cnt**2).sum()/n**2)
    return g
def expect(f, clean, parts, C, T):
    L = len(clean); E = 0.0
    for z in itertools.product(range(C), repeat=L):
        E += np.prod([T[clean[i], z[i]] for i in range(L)]) * f(z, parts, C)
    return E
C = 3
clean = [0,0,0,1,1,1,2,2]
parts = {
 'coarse(2 balls, mixed)':      [[0,1,2,3],[4,5,6,7]],
 'good split(3, purer)':        [[0,1,2],[3,4,5],[6,7]],
 'fragments(singletons)':       [[0],[1],[2],[3],[4],[5],[6],[7]],
 'bad split(4, mixed pairs)':   [[0,3],[1,4],[2,6],[5,7]],
}
for noise, Tf in [('symmetric', T_sym), ('pair', T_pair)]:
    for eta in [0.0, 0.4]:
        T = Tf(C, eta)
        row = []
        for k, P in parts.items():
            row.append((k, expect(G_part, clean, P, C, T), expect(U_part, clean, P, C, T)))
        print(f"--- {noise} eta={eta}")
        for k, g, u in row: print(f"   {k:28s} plug-in Gini={g:.3f}   unbiased pairwise={u:.3f}")
