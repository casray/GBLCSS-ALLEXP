import itertools, numpy as np
def T_sym(C, eta):
    T = np.full((C, C), eta/(C-1)); np.fill_diagonal(T, 1-eta); return T
def u_hat(labels, C):
    n = len(labels); cnt = np.bincount(labels, minlength=C)
    return 1 - (cnt*(cnt-1)).sum()/(n*(n-1))
def score(z, cut, parent_of, C, mode):
    """sum_B n_B * U_B over leaves; leaves with <2 labels: 'prior' -> 1-1/C, 'inherit' -> nearest ancestor with >=2 labels"""
    s = 0.0
    for B in cut:
        if len(B) >= 2: s += len(B)*u_hat([z[i] for i in B], C)
        elif len(B) == 1:
            if mode == 'prior': s += (1-1/C)
            else:
                A = parent_of[tuple(B)]
                while len(A) < 2: A = parent_of[tuple(A)]
                s += u_hat([z[i] for i in A], C)
    return s
def expect(clean, cut, parent_of, C, T, mode):
    L = len(clean); E = 0.0
    for z in itertools.product(range(C), repeat=L):
        E += np.prod([T[clean[i], z[i]] for i in range(L)]) * score(z, cut, parent_of, C, mode)
    return E
C = 3
# counterexample 1: a 2-label mixed ball split into two singletons
clean = [0, 1]; root = [0, 1]; parent = {(0,): root, (1,): root}
for mode in ['prior', 'inherit']:
    print(mode, 'root=', score(clean, [root], parent, C, mode), ' two singletons=', score(clean, [[0],[1]], parent, C, mode))
# fixed-partition expectation identity with inheritance (tree: root -> A,B ; A -> A1,A2 singleton-ish)
clean = [0,0,1,1,2,0,1]
root=[0,1,2,3,4,5,6]; A=[0,1,2,5]; B=[3,4,6]; A1=[0,5]; A2=[1]; A3=[2]
parent={(0,1,2,5):root,(3,4,6):root,(0,5):A,(1,):A,(2,):A}
cuts={'root':[root],'A|B':[A,B],'A1|A2|A3|B':[A1,A2,A3,B]}
for eta in [0.0,0.3,2/3]:
    gam=1-C*eta/(C-1); T=T_sym(C,eta)
    out=[]
    for k,cut in cuts.items():
        En=expect(clean,cut,parent,C,T,'inherit'); Ec=score(clean,cut,parent,C,'inherit')
        pred=gam**2*Ec+(1-gam**2)*(1-1/C)*len(clean)
        out.append(f"{k}: E={En:.4f} pred={pred:.4f}")
    print(f"eta={eta:.3f} | "+" | ".join(out))
