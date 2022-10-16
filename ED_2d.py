#import numpy as jnp
import jax
import jax.numpy as jnp

class Hubbard_2d_ED(object):
    def __init__(self, Lx, Ly, N, t, U):
        self.Lx = Lx # number of sites in x direction
        self.Ly = Ly 
        self.Lsite = Lx * Ly 
        self.N = N # number of particles of each spin
        self.t = t # hopping amptitude
        self.U = U # on-site repulsive potential
        self.idx = {} # { index of site: (x, y) }
        self.idx_inv = {} # { (x, y): index of site }
        for i in range(Lx):
            for j in range(Ly):
                parity = i%2
                base = i*Ly
                if parity == 0:
                    self.idx[base + j] = (i, j)
                    self.idx_inv[(i, j)] = base + j
                else:
                    self.idx[base + (Ly-j) - 1] = (i, j)
                    self.idx_inv[(i, j)] = base + (Ly-j) - 1
        
    def get_basis(self):
        from itertools import combinations
        up = combinations(range(self.Lx * self.Ly), self.N)
        up = [ sorted(list(xx)) for xx in up ]
        up_basis = up
        #for i, occ in enumerate(up):
        #    #tmp = jnp.zeros(L**2)
        #    #tmp[list(occ)] = 1
        #    #up_basis.append(occ)
        down_basis = up_basis
        basis = []
        for x in up_basis:
            for y in down_basis:
                basis.append([x, y])
                #print(x, y)
        return basis  

    def get_T(self):
        basis = self.get_basis()
        
        def hopping(state, x_and_y, dire): 
            state_new = state.copy()
            parity = None
            x, y = x_and_y[0], x_and_y[1]
            x1, y1 = (x+dire[0])%self.Lx, (y+dire[1])%self.Ly
            state_coordinates = { i: self.idx[i] for i in state_new } # { idx: (x, y)  }
            
            if ( (x, y) not in state_coordinates.values() ) or \
                          ( (x1, y1) in state_coordinates.values() ):
                state_new = None
                parity = None
            else:
                # the hopping operator is given by C^{\dagger}_{\alpha} C_{\beta}
                alpha = self.idx_inv[(x1, y1)]
                beta = self.idx_inv[(x, y)]
                l = state_new.index(beta) + 1
                parity1 = l-1
                state_new.remove(beta)
                state_new = sorted(state_new + [alpha])
                s = state_new.index(alpha) + 1
                parity2 = s-1 
                import math
                parity = int(math.pow(-1, parity1 + parity2))
            return state_new, parity

        def all_hoppings(state):
            up, down = state[0], state[1]
            hopped_up, hopped_down = [], []
            for x in range(self.Lx):
                for y in range(self.Ly):
                    dires = [ (1,0), (-1,0), (0,1), (0,-1) ]
                    for dire in dires:
                        up_new, up_sign = hopping(up, (x,y), dire)
                        down_new, down_sign = hopping(down, (x,y), dire)
                        tmp_up = ( [ up_new, down ], up_sign )
                        tmp_down = ( [ up, down_new ], down_sign )
                        if tmp_up not in hopped_up:
                            hopped_up.append(tmp_up)
                        if tmp_down not in hopped_down:
                            hopped_down.append(tmp_down)
            state_hopped = hopped_up + hopped_down
            return state_hopped
        """
        test hopping and all_hoppings
        #return hopping, all_hoppings
        state = basis[0]
        up, down = state[0], state[1]
        print('state:', up, down)
        #return hopping(up, (0, 2), (-1,0))
        return all_hoppings(state)  
        """
        overlap = {} 
        for ib, bas in enumerate(basis):
            bas_hopped = all_hoppings(bas)
            matched = []
            for xx in bas_hopped:
                (state_new, sign) = xx
                for ibb, bass in enumerate(basis):
                    if bass == state_new:
                        matched.append((ibb, sign))
            overlap[ib] = matched
        T_matr = jnp.zeros((len(basis), len(basis)))

        for ii in overlap:
            for jj in overlap[ii]:
                #T_matr[ii][jj[0]] = self.t * jj[1]
                T_matr = T_matr.at[ii,jj[0]].set(self.t * jj[1])

        #return T_matr, overlap 
        return T_matr
 
    def get_U(self):
        basis = self.get_basis()
        def count_double_occ(state): # count the number of double occupation 
            return len( [x for x in state[0] if x in state[1]] )

        U_matr = jnp.zeros((len(basis), len(basis)))
        for ib, ba in enumerate(basis):
            #U_matr[ib][ib] = count_double_occ(ba) * self.U
            U_matr = U_matr.at[ib, ib].set(count_double_occ(ba) * self.U)
        return U_matr

    def get_Hamiltonian(self):
        #T_matr, T_dic = self.get_T()
        T_matr = self.get_T()
        U_matr = self.get_U()
        return T_matr + U_matr

    def diagonalize(self):
        hamiltonian = self.get_Hamiltonian()
        eigvals, eigvecs = jnp.linalg.eigh(hamiltonian)
        return eigvals, eigvecs

    def free_energy(self, beta):
        E, _ = self.diagonalize()
        F = 0.
        Z = sum([ jnp.exp(-beta*ee) for ee in E ])
        for i in range(len(E)):
            p = jnp.exp(-beta * E[i]) / Z
            f = p * (1/beta * jnp.log(p) + E[i])
            F += f
        return F
    
    
