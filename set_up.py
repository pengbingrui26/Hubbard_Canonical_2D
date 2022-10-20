import jax
import jax.numpy as jnp
from free_model import Hubbard_2d_free
from ED_2d import Hubbard_2d_ED
from functools import partial
import time 

jax.config.update("jax_enable_x64", True)

I = complex(0., 1.)

# ================================================================================================

def init_fn(model):
    #Tmatr = model.get_Hfree_half()

    def _make_expMF(elements):
        # elements is a 1D array of shape (model.Lsite * model.Lsite, )
        """
        diags = elements[:model.Lsite]
        off_diags = elements[model.Lsite:]
        MF = jnp.zeros((model.Lsite, model.Lsite))
        ntotal = 0
        for i in range(model.Lsite-1):
            MF = MF.at[i, i+1:].set(off_diags[ntotal: ntotal + model.Lsite - i - 1])
            ntotal += (model.Lsite - i - 1)
        MF = MF + MF.T
        for i in range(model.Lsite):
            MF = MF.at[i, i].set(diags[i])
        """
        elements_matr = jnp.reshape(elements, (model.Lsite, model.Lsite))
        MF = elements_matr + elements_matr.T

        expMF = jax.scipy.linalg.expm(-I * MF)

        #expMF_conjugate = jnp.conjugate(expMF.T)
        #print('check unitary:')
        #print(jnp.dot(expMF_conjugate, expMF))

        return expMF


    # act on spin SD
    def _make_expU(spin, sigma, tau):
        # exp(-i*V*tau) = (1/2)^L * exp{-\tau*U*N/2} * \sum_{{sigma}} exp{alpha \sum_i \sigma_i (n_{i,up} - n_{i,down})}
        # alpha = arccosh(exp{i*tau*U/2})
        Gamma = jnp.exp(-I*tau*model.U*model.N*2/2) 
        alpha = jnp.arccosh(jnp.exp(I*tau*model.U/2))

        #nspin_arr = jnp.hstack((jnp.ones(model.Lsite), -jnp.ones(model.Lsite)))
        nspin_arr = spin * jnp.ones(model.Lsite)
        V_diags = jnp.multiply(sigma, nspin_arr)
        expU_diags = jnp.power(0.5, model.Lsite) * Gamma * jnp.exp(alpha * V_diags)

        return expU_diags


    # act on spin SD
    #@partial(jax.jit)
    def _evolve(psi0, spin, sigmas, params):
        # sigmas has the shape (nlayer, model.L)
        # params = [nelement, Utaus ] * nlayer
        #nelement = int(model.Lsite * (model.Lsite + 1)/2)
        nelement = int(model.Lsite ** 2)
        nparam_per_layer = nelement + 1
        #elements = params[:int(model.Lsite * (model.Lsite + 1)/2)]
        #taus = params[int(model.Lsite * (model.Lsite + 1)/2):]
        #lenth = taus.shape[-1]
        nlayer = int(params.shape[0]/nparam_per_layer)
        psi = psi0
        #print('psi.shape:', psi.shape)

        """
        def body_fun(ilayer, psi):
            elements = params[ilayer*nparam_per_layer: ilayer*nparam_per_layer + nelement]
            Utau = params[ilayer*nparam_per_layer + nelement]
            sigma = sigmas[ilayer]

            #expT = jax.lax.pow(expT0, tau2.astype(complex))
            #expT = jax.scipy.linalg.expm(-I * Tmatr * tau2)
            expMF = _make_expMF(elements)
            expU_diag = _make_expU(spin, sigma, Utau)

            psi = jnp.multiply(expU_diag, psi.T).T
            psi = jnp.dot(expMF, psi)

            return jnp.dot(expMF, jnp.multiply(expU_diag, psi.T).T)

        psi = jax.lax.fori_loop(0, nlayer, body_fun, psi.astype('complex'))
        """

        for ilayer in range(nlayer):
            elements = params[ilayer*nparam_per_layer: ilayer*nparam_per_layer + nelement]
            Utau = params[ilayer*nparam_per_layer + nelement]

            sigma = sigmas[ilayer]
            #expT = jax.lax.pow(expT0, tau2.astype(complex))
            #expT = jax.scipy.linalg.expm(-I * Tmatr * tau2)
            expMF = _make_expMF(elements)
            expU_diag = _make_expU(spin, sigma, Utau)

            #psi = jnp.matmul(expMF, psi)
            psi = jnp.multiply(expU_diag, psi.T).T
            psi = jnp.matmul(expMF, psi)

        return psi

    # act on full SD
    #@partial(jax.jit)
    def make_W(psi0, sigmas_long, params):
        # sigmas_long has the shape of (nlayer, model.Lsite*2) 
        #print('psi0.shape:', psi0.shape)

        sigmasL, sigmasR = sigmas_long[:, :model.Lsite], sigmas_long[:, model.Lsite:]
        psi0_up, psi0_down = psi0[:model.Lsite, :], psi0[model.Lsite:, :]

        psi_up_L = _evolve(psi0_up, 1, sigmasL, params)
        psi_down_L = _evolve(psi0_down, -1, sigmasL, params)

        psi_up_R = _evolve(psi0_up, 1, sigmasR, params)
        psi_down_R = _evolve(psi0_down, -1, sigmasR, params)

        #psi_L = jnp.vstack((psi_up_L, psi_down_L))
        #psi_R = jnp.vstack((psi_up_R, psi_down_R))

        W_up = jnp.dot(jnp.conjugate(psi_up_L.T), psi_up_R)
        W_down = jnp.dot(jnp.conjugate(psi_down_L.T), psi_down_R)

        W = W_up + W_down
        W = jnp.linalg.det(W)

        return W.real, W.imag


    # act on full SD
    def make_Eloc_no_evolve(psiL, psiR, taus):
        psiL_up, psiL_down = psiL[:model.Lsite, :], psiL[model.Lsite:, :]
        psiR_up, psiR_down = psiR[:model.Lsite, :], psiR[model.Lsite:, :]
        
        #S = jnp.dot(jnp.conjugate(psiL.T), psiR)
        S = jnp.dot(jnp.conjugate(psiL_up.T), psiR_up) + jnp.dot(jnp.conjugate(psiL_down.T), psiR_down)
        S_inv = jnp.linalg.pinv(S)

        Sup = jnp.dot(jnp.conjugate(psiL_up.T), psiR_up)
        Sup_inv = jnp.linalg.pinv(Sup)
        Sdown = jnp.dot(jnp.conjugate(psiL_down.T), psiR_down)
        Sdown_inv = jnp.linalg.pinv(Sdown)

        Tmatr = model.get_Hfree()
        T_loc = jnp.dot(jnp.conjugate(psiL.T), jnp.dot(Tmatr, psiR))
        T_loc = jnp.trace(jnp.dot(S_inv, T_loc))

        U_loc = 0.
        for k in range(model.Lsite):
            psiL_up_k = jnp.conjugate(psiL_up)[k,:]
            psiR_up_k = psiR_up[k,:]
            A_up = jnp.outer(psiL_up_k, psiR_up_k)
            k_spin_up = jnp.trace(jnp.dot(Sup_inv, A_up))
         
            psiL_down_k = jnp.conjugate(psiL_down)[k,:]
            psiR_down_k = psiR_down[k,:]
            A_down = jnp.outer(psiL_down_k, psiR_down_k)
            k_spin_down = jnp.trace(jnp.dot(Sdown_inv, A_down))

            U_loc += k_spin_up * k_spin_down

        Eloc = jnp.linalg.det(S) * (T_loc + model.U * U_loc)
        return Eloc.real, Eloc.imag

    # act on full SD
    def make_Eloc_old(psi0, sigmas_long, params):
        # sigmas_long has shape (nlayer, model.Lsite*2)
        sigmasL = sigmas_long[:, :model.Lsite]
        sigmasR = sigmas_long[:, model.Lsite:]

        psi0_up, psi0_down = psi0[:model.Lsite, :], psi0[model.Lsite:, :]
        
        psiL_up = _evolve(psi0_up, 1, sigmasL, params)
        psiL_down = _evolve(psi0_down, -1, sigmasL, params)

        psiR_up = _evolve(psi0_up, 1, sigmasR, params)
        psiR_down = _evolve(psi0_down, -1, sigmasR, params)

        psiL = jnp.vstack((psiL_up, psiL_down))
        psiR = jnp.vstack((psiR_up, psiR_down))

        #S = jnp.dot(jnp.conjugate(psiL.T), psiR)
        S = jnp.dot(jnp.conjugate(psiL_up.T), psiR_up) + jnp.dot(jnp.conjugate(psiL_down.T), psiR_down)
        S_inv = jnp.linalg.pinv(S)

        Sup = jnp.dot(jnp.conjugate(psiL_up.T), psiR_up)
        Sup_inv = jnp.linalg.pinv(Sup)
        Sdown = jnp.dot(jnp.conjugate(psiL_down.T), psiR_down)
        Sdown_inv = jnp.linalg.pinv(Sdown)

        Tmatr = model.get_Hfree()
        T_loc = jnp.dot(jnp.conjugate(psiL.T), jnp.dot(Tmatr, psiR))
        T_loc = jnp.trace(jnp.dot(S_inv, T_loc))

        U_loc = 0. + 1j * 0.

        def body_fun(k, uloc):
            psiL_up_k = jnp.conjugate(psiL_up)[k,:]
            psiR_up_k = psiR_up[k,:]
            A_up = jnp.outer(psiL_up_k, psiR_up_k)
            k_spin_up = jnp.trace(jnp.dot(Sup_inv, A_up))
         
            psiL_down_k = jnp.conjugate(psiL_down)[k,:]
            psiR_down_k = psiR_down[k,:]
            A_down = jnp.outer(psiL_down_k, psiR_down_k)
            k_spin_down = jnp.trace(jnp.dot(Sdown_inv, A_down))

            return uloc + k_spin_up * k_spin_down

        for k in range(model.Lsite):
            psiL_up_k = jnp.conjugate(psiL_up)[k,:]
            psiR_up_k = psiR_up[k,:]
            A_up = jnp.outer(psiL_up_k, psiR_up_k)
            k_spin_up = jnp.trace(jnp.dot(Sup_inv, A_up))
         
            psiL_down_k = jnp.conjugate(psiL_down)[k,:]
            psiR_down_k = psiR_down[k,:]
            A_down = jnp.outer(psiL_down_k, psiR_down_k)
            k_spin_down = jnp.trace(jnp.dot(Sdown_inv, A_down))

            U_loc += k_spin_up * k_spin_down

        #U_loc = jax.lax.fori_loop(0, model.Lsite, body_fun, U_loc)

        Eloc = jnp.linalg.det(S) * (T_loc + model.U * U_loc)
        return Eloc.real, Eloc.imag

    def make_Eloc(psi0, sigmas_long, params):
        # sigmas_long has shape (nlayer, model.Lsite*2)
        sigmasL = sigmas_long[:, :model.Lsite]
        sigmasR = sigmas_long[:, model.Lsite:]

        psi0_up, psi0_down = psi0[:model.Lsite, :], psi0[model.Lsite:, :]
        
        psiL_up = _evolve(psi0_up, 1, sigmasL, params)
        psiL_down = _evolve(psi0_down, -1, sigmasL, params)

        psiR_up = _evolve(psi0_up, 1, sigmasR, params)
        psiR_down = _evolve(psi0_down, -1, sigmasR, params)

        psiL = jnp.vstack((psiL_up, psiL_down))
        psiR = jnp.vstack((psiR_up, psiR_down))

        #S = jnp.dot(jnp.conjugate(psiL.T), psiR)
        S = jnp.dot(jnp.conjugate(psiL_up.T), psiR_up) + jnp.dot(jnp.conjugate(psiL_down.T), psiR_down)
        S_inv = jnp.linalg.pinv(S)

        Sup = jnp.dot(jnp.conjugate(psiL_up.T), psiR_up)
        Sup_inv = jnp.linalg.pinv(Sup)
        Sdown = jnp.dot(jnp.conjugate(psiL_down.T), psiR_down)
        Sdown_inv = jnp.linalg.pinv(Sdown)

        Tmatr = model.get_Hfree()
        T_loc = jnp.dot(jnp.conjugate(psiL.T), jnp.dot(Tmatr, psiR))
        T_loc = jnp.trace(jnp.dot(S_inv, T_loc))

        #U_loc = 0. + 1j * 0.

        make_outer_vmapped = jax.vmap(jnp.outer, in_axes = (0, 0), out_axes = 0)
        make_matmul_vmapped = jax.vmap(jnp.matmul, in_axes = (None, 0), out_axes = 0)
        make_trace_vmapped = jax.vmap(jnp.trace, in_axes = 0, out_axes = 0)

        A_up_all = make_outer_vmapped(jnp.conjugate(psiL_up), psiR_up)
        k_spin_up_all = make_matmul_vmapped(Sup_inv, A_up_all)
        k_spin_up = make_trace_vmapped(k_spin_up_all)

        A_down_all = make_outer_vmapped(jnp.conjugate(psiL_down), psiR_down)
        k_spin_down_all = make_matmul_vmapped(Sdown_inv, A_down_all)
        k_spin_down = make_trace_vmapped(k_spin_down_all)

        U_loc = jnp.multiply(k_spin_up, k_spin_down)
        U_loc = U_loc.sum()

        Eloc = jnp.linalg.det(S) * (T_loc + model.U * U_loc)
        return Eloc.real, Eloc.imag


    # act on full SD
    def make_eloc(psi0, sigmas_long, taus):
        # sigmas_long has shape (nlayer, model.Lsite*2)
        Eloc_real, Eloc_imag = make_Eloc(psi0, sigmas_long, taus)
        Eloc = Eloc_real + Eloc_imag * I

        W_real, W_imag = make_W(psi0, sigmas_long, taus)
        W = W_real + W_imag * I

        eloc = Eloc / W
        return eloc.real, eloc.imag

    return make_W, make_Eloc, make_eloc
    #return _make_expMF
    #return _evolve
    #return make_Eloc_new
    
   

# ========================================================================== 

def flip(sigmas_long, key):
    # sigmas_long has shape (nlayer, model.Lsite*2)
    Lsite_full = sigmas_long.shape[-1]
    sigmas_reshaped = sigmas_long.flatten()
    L_all = sigmas_reshaped.shape[-1]
    isite = jax.random.choice(key = key, a = jnp.arange(L_all))
    sigmas_reshaped = sigmas_reshaped.at[isite].set(-sigmas_reshaped[isite]) 
    sigmas_new = sigmas_reshaped.reshape(-1, Lsite_full)
    return sigmas_new

flip_vmapped = jax.vmap(flip, in_axes = (0, 0), out_axes = 0)


def random_init_sigma(Lsite_full, nlayer, key):
    return jax.random.choice(key = key, a = jnp.array([1, -1]), shape = (nlayer, Lsite_full)) 

#random_init_sigma_vmapped = jax.vmap(random_init_sigma, in_axes = (None, None, 0), out_axes = 0)


def random_init_sigma_vmapped(batch, nlayer, Lsite_full, key):
    sigma_vmapped = jax.random.choice(key = key, a = jnp.array([1, -1]), \
                                        shape = (batch, nlayer, Lsite_full)) 
    return sigma_vmapped
   



# test ====================================================================

def test_make_expMF():
    Lx, Ly = 2, 2
    Lsite = Lx * Ly
    N = int(Lsite/2)
    t, U = 1., 0.
    model = Hubbard_2d_free(Lx, Ly, N, t, U)
    make_expMF = init_fn(model)
    #n = model.Lsite * (model.Lsite + 1) /2 
    n = model.Lsite**2
    import numpy as np
    elements = np.random.rand(int(n))
    print('elements:')
    print(elements)
    _ = make_expMF(elements)


def make_direct_nlayer(L, N, t, U, taus):
    model = Hubbard_2d_ED(L, N, t, U)
    Hmatr = model.get_T()
    Tmatr = model.get_U()
    Umatr = model.get_Hamiltonian()
 
    model_free = Hubbard_2d_ED(L, N, t, U)
    _, psi = model_free.eigs()
    #psi0 = psi[:, 0]
    #print(psi0)
    #print('psi0 norm:')
    #print(jnp.linalg.norm(psi0))

    lenth = taus.shape[-1]
    nlayer = int(lenth/2)

    Es = []
    for i in range(psi.shape[-1]):
        psi0 = psi[:, i]
        for ilayer in range(nlayer):
            tau1 = taus[lenth - ilayer*2 - 1]
            tau2 = taus[lenth - ilayer*2 - 2]
            expT = jax.scipy.linalg.expm(-I * Tmatr * tau2)
            expU = jax.scipy.linalg.expm(-I * Umatr * tau1)
            psi0 = jnp.dot(expT, jnp.dot(expU, psi0))
 
        E = jnp.dot(jnp.conjugate(psi0.T), jnp.dot(Hmatr, psi0))
        E = E / jnp.dot(jnp.conjugate(psi0.T), psi0)
        Es.append(E)

    return Es


def test_evolve():
    Lx, Ly = 2, 2
    N = int(Lx * Ly/2)
    t, U = 1., 2.
    model = Hubbard_2d_free(Lx, Ly, N, t, U)
    psi0_half = model.get_psi0_half()
    psi0 = psi0_half[0]
    print('psi0.shape:', psi0.shape)
    evolve = init_fn(model)
    spin = 1
    sigmas = jnp.array([[1, -1, 1, -1]]) #sigmas has the shape (nlayer, model.Lsite)
    nelement = int(model.Lsite * (model.Lsite + 1) / 2)
    params = jnp.array([0.1] * nelement)
    psi = evolve(psi0, spin, sigmas, params)
    print(psi)


def test_make_W():
    L = 2
    N = int(L/2)
    t, U = 1., 2.
    model = Hubbard_2d_free(L, N, t, U)
    psi0 = model.get_ground_state()

    psi_norm = jnp.linalg.det(jnp.dot(jnp.conjugate(psi0.T), psi0))
 
    sigma_long = jnp.array([1, -1, -1, 1])
    taus = jnp.array([1., 1.])

    make_W, _ = init_fn(model)
    W = make_W(psi0, sigma_long, taus)

    print("W:", W)


def test_make_Eloc():
    Lx, Ly = 2, 2
    Lsite = Lx * Ly
    N = int(Lsite/2)
    t, U = 1., 0.
    model = Hubbard_2d_free(Lx, Ly, N, t, U)
    psi0 = model.get_ground_state()

    #psi_norm = jnp.linalg.det(jnp.dot(jnp.conjugate(psi0.T), psi0))
    #print("psi_norm:")
    #print(psi_norm)
 
    make_Eloc = init_fn(model)
    sigmas_long = jnp.array([[1, -1, 1, -1, 1, -1, 1, -1]]) 
    #sigmas_long has the shape (nlayer, model.Lsite * 2)
    nelement = int(model.Lsite * (model.Lsite + 1) / 2)
    params = jnp.array([0.1] * nelement)
    Eloc = make_Eloc(psi0, sigmas_long, params)
    print("Eloc:", Eloc)


def test_flip():
    sigma = jnp.array([1, 1, 1, 1])
    key = jax.random.PRNGKey(42)

    """
    for i in range(20):
        key_old, key = jax.random.split(key, 2)
        print(key)
        sigma_new = flip(sigma, key)
        print(sigma_new)
    """

    sigma_vmapped = jnp.array([ [1, 1], [-1, -1], [1, -1] ])
    key_vmapped = jax.random.split(key, 3)
    sigma_vmapped_new = flip_vmapped(sigma_vmapped, key_vmapped)
    print(sigma_vmapped_new)  
 

def test_random_init_sigma():
    L = 2
    L2 = L * 2
    key = jax.random.PRNGKey(42)
    sigma_init = random_init_sigma(L2, key)
    print(sigma_init)

    batch = 5
    key_vmapped = jax.random.split(key, batch)
    sigma_init_vmapped = random_init_sigma_vmapped(L2, key_vmapped)
    print(sigma_init_vmapped)


def test_make_direct():
    t, U = 1., 2.5
    tau1, tau2 = 1., 1.
    E = make_direct_nlayer(t, U, tau1, tau2)
    print(E)



# run ======================================================================

#test_make_expMF()
#test_make_expU()
#test_evolve()
#test_evolve_nlayer()
#test_make_W()
#test_make_Eloc()
#test_random_init_sigma()
#test_flip()
#test_make_direct()
#test_metropolis()
#test_metropolis_vmapped()



