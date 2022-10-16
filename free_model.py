#import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

class Hubbard_2d_free(object):
    
    def __init__(self, Lx, Ly, N, t, U):
        self.Lx = Lx
        self.Ly = Ly
        self.N = N
        self.t = t
        self.U = U
        self.Lsite = self.Lx * self.Ly
        ##assert self.N == int(self.Lsite/2)

        self.idx = {} # { index of site: (x, y) }
        self.idx_inv = {} # { (x, y): index of site }
        for i in range(self.Lx):
            for j in range(self.Ly):
                parity = i%2
                base = i*self.Ly
                if parity == 0:
                    self.idx[base + j] = (i, j)
                    self.idx_inv[(i, j)] = base + j
                else:
                    self.idx[base + (self.Ly-j) - 1] = (i, j)
                    self.idx_inv[(i, j)] = base + (self.Ly-j) - 1
        self.idx_up = self.idx
        self.idx_down = { idx+self.Lsite: self.idx[idx] for idx in self.idx }
        self.idx_inv_up = self.idx_inv
        self.idx_inv_down = { x_and_y: self.idx_inv[x_and_y]+self.Lsite \
                              for x_and_y in self.idx_inv }

        self.boundary_idx = []
        for idx in self.idx:
            (x, y) = self.idx[idx]
            if (x in [0, self.Lx - 1]) or (y in [0, self.Ly - 1]):
                self.boundary_idx.append(idx)
                #self.boundary_idx.append(idx+self.Lsite)

        H_free_half = jnp.zeros((self.Lsite, self.Lsite))
        #for upsite in self.idx_up:
        for upsite in range(self.Lsite):
            (x, y) = self.idx_up[upsite]
            dires = [ (1, 0), (-1, 0), (0, 1), (0, -1) ]
            for dire in dires:
                x1, y1 = (x + dire[0]) % self.Lx, (y + dire[1]) % self.Ly
                upsite_new = self.idx_inv[(x1, y1)]
                #if upsite in self.boundary_idx:
                #    H_free_half[upsite][upsite_new] += self.t
                #elif upsite not in self.boundary_idx:
                H_free_half = H_free_half.at[upsite, upsite_new].set(-self.t)

        H_free = jnp.zeros((self.Lsite*2, self.Lsite*2))
        H_free = H_free.at[:self.Lsite, :self.Lsite].set(H_free_half)
        H_free = H_free.at[self.Lsite:, self.Lsite:].set(H_free_half)

        self.H_free_half = H_free_half
        self.H_free = H_free

    def get_Hfree_half(self):
        return self.H_free_half

    def get_Hfree(self):
        return self.H_free
           
    def get_eigs_half(self):
        H_free_half = self.H_free_half
        eigvals, eigvecs = jnp.linalg.eigh(H_free_half)
        sorted_idx = jnp.argsort(eigvals)
        eigvals = eigvals[sorted_idx]
        eigvecs = eigvecs[:, sorted_idx]
        return eigvals, eigvecs

    def get_psi0_half(self):
        _, eigvecs_half = self.get_eigs_half()
        psi0_set = []
        npsi = eigvecs_half.shape[-1]
        import itertools
        list_idx = list(itertools.combinations(range(npsi), int(npsi/2)))
        for idx in list_idx:
            idx = list(idx)
            psi = eigvecs_half[:, idx]
            psi0_set.append(psi)
        return psi0_set

    def get_ground_energy(self):
        eigvals_half, eigvecs_half = self.get_eigs_half()
        return jnp.sum(eigvals_half[:self.N]) * 2

    def get_ground_state(self):
        _, eigvecs_half = self.get_eigs_half()
        psi_up = eigvecs_half[:, :self.N]
        psi_down = eigvecs_half[:, :self.N]

        psi_full = jnp.zeros((self.Lsite*2, self.N*2))
        psi_full = psi_full.at[:self.Lsite, :self.N].set(psi_up)
        psi_full = psi_full.at[self.Lsite:, self.N:].set(psi_down)
        return psi_full 

    def get_psi0_full(self):
        psi0_full_set = []
        psi0_half_set = self.get_psi0_half()
        n_psi0 = len(psi0_half_set)
        for i in range(n_psi0):
            for j in range(n_psi0):
                psi0_up = psi0_half_set[i]                
                psi0_down = psi0_half_set[j]
                #psi0_full = jnp.vstack((psi0_up, psi0_down))                
                psi0_full = jnp.zeros((self.Lsite*2, self.N*2))
                psi0_full = psi0_full.at[:self.Lsite, :self.N].set(psi0_up)
                psi0_full = psi0_full.at[self.Lsite:, self.N:].set(psi0_down)
                psi0_full_set.append(psi0_full)
        return psi0_full_set

    def get_psi0_nset(self, n_psi0):
        psi0_full_set = []
        psi0_half_set = self.get_psi0_half()
        npsi0_half = len(psi0_half_set)
        stop = False
        for i in range(npsi0_half):
            if stop == True:
                break
            for j in range(npsi0_half):
                if len(psi0_full_set) == n_psi0:
                    stop = True
                    break
                psi0_up = psi0_half_set[i]                
                psi0_down = psi0_half_set[j]
                #psi0_full = jnp.vstack((psi0_up, psi0_down))                
                psi0_full = jnp.zeros((self.Lsite*2, self.N*2))
                psi0_full = psi0_full.at[:self.Lsite, :self.N].set(psi0_up)
                psi0_full = psi0_full.at[self.Lsite:, self.N:].set(psi0_down)
                psi0_full_set.append(psi0_full)
        return psi0_full_set


# =========================================================
    def get_basis(self):
        from itertools import combinations
        up = combinations(range(self.Lsite), self.N)
        up = [ sorted(list(xx)) for xx in up ]
        up_basis = []
        for i, occ in enumerate(up):
            up_basis.append(occ)
        down_basis = up_basis
        basis = []
        for x in up_basis:
            for y in down_basis:
                y_new = [ (yy+self.Lsite) for yy in y ] 
                basis.append(x+y_new)
        return basis  

    
