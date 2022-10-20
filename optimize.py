import jax
import jax.numpy as jnp
from free_model import Hubbard_2d_free 
from ED_2d import Hubbard_2d_ED
import time 
from functools import partial

from set_up import init_fn, \
                   flip, flip_vmapped, random_init_sigma, random_init_sigma_vmapped           

 
jax.config.update("jax_enable_x64", True)

I = complex(0., 1.)

# =============================================================================

# sample W and W_sign
def sample_simple(model, psi0, key, batch, taus, nthermal, nsample, ninterval):
    nlayer = int(taus.shape[-1]/2)
 
    make_W, _, _ = init_fn(model)
    make_W_vmapped = jax.vmap(make_W, in_axes = (None, 0, None), out_axes = (0, 0))

    key_init, key_flip = jax.random.split(key, 2)
    sigma = random_init_sigma_vmapped(batch, nlayer, model.Lsite*2, key_init)  # shape: (batch, nlayer, model.Lsite*2)

    sigma_sampled = []
    W_sampled = []
    sign_sampled = []

    start = time.time()

    W_real, W_imag = make_W_vmapped(psi0, sigma, taus)
    W = W_real + I * W_imag
    W_norm = abs(W)
    W_sign = W / W_norm

    for imove in range(nthermal + nsample * ninterval):
        print("imove:", imove)

        if (imove > nthermal) and ((imove - nthermal) % ninterval == 0):
            sigma_sampled.append(sigma)
            W_sampled.append(W)
            sign_sampled.append(W_sign)
        
        ## flip
        key_uniform, key_proposal, key_flip = jax.random.split(key_flip, 3)
        key_proposal = jax.random.split(key_proposal, batch)

        sigma_proposal = flip_vmapped(sigma, key_proposal)

        W_proposal_real, W_proposal_imag = make_W_vmapped(psi0, sigma_proposal, taus)
        W_proposal = W_proposal_real + I * W_proposal_imag
        W_proposal_norm = abs(W_proposal)

        #W_ratio = abs(W_proposal) / abs(W)
        W_ratio = W_proposal_norm / W_norm
        ratio = jnp.where(W_ratio < 1., W_ratio, 1.)
        proposal = jax.random.uniform(key_uniform, shape = (batch, ))

        accept = ratio > proposal
        sigma = jnp.where(accept[:, None, None], sigma_proposal, sigma)

        W = jnp.where(accept, W_proposal, W)
        W_norm = abs(W)
        W_sign = W / W_norm
        ##

    end = time.time()
    print("time for MCMC:", end - start)

    sigma_sampled = jnp.array(sigma_sampled)
    sigma_sampled = jnp.concatenate(sigma_sampled, axis =0)
    W_sampled = jnp.array(W_sampled).flatten()
    sign_sampled = jnp.array(sign_sampled).flatten()

    return sigma_sampled, W_sampled, sign_sampled


def make_En_and_grad(model, psi0, key, batch, taus, nthermal, nsample, ninterval):
    make_W, make_Eloc, make_eloc = init_fn(model)
    make_Eloc_vmapped = jax.vmap(make_Eloc, in_axes = (None, 0, None), out_axes = (0, 0))

    make_grad_W = jax.jacrev(make_W, argnums = -1)
    make_grad_Eloc = jax.jacrev(make_Eloc, argnums = -1)
    make_grad_eloc = jax.jacrev(make_eloc, argnums = -1)

    make_grad_W_vmapped = jax.vmap(make_grad_W, in_axes = (None, 0, None), out_axes = (0, 0))
    make_grad_Eloc_vmapped = jax.vmap(make_grad_Eloc, in_axes = (None, 0, None), out_axes = (0, 0))
    make_grad_eloc_vmapped = jax.vmap(make_grad_eloc, in_axes = (None, 0, None), out_axes = (0, 0))

    sigma, W, sign = sample_simple(model, psi0, key, batch, taus, nthermal, nsample, ninterval)

    start = time.time()

    Eloc_real, Eloc_imag = make_Eloc_vmapped(psi0, sigma, taus)
    Eloc = Eloc_real + I * Eloc_imag
    
    eloc = Eloc / W

    sign_dot_eloc = jnp.multiply(sign, eloc)
    E_mean = sign_dot_eloc.sum() / sign.sum()

    grad_W_real, grad_W_imag = make_grad_W_vmapped(psi0, sigma, taus)
    grad_W = grad_W_real + I * grad_W_imag
 
    grad_Eloc_real, grad_Eloc_imag = make_grad_Eloc_vmapped(psi0, sigma, taus)
    grad_Eloc = grad_Eloc_real + I * grad_Eloc_imag
    
    grad_eloc = (jnp.multiply(grad_Eloc.T, W) - jnp.multiply(Eloc, grad_W.T)) / jnp.multiply(W, W)

    O = grad_W.T / W

    dominator = jnp.multiply( sign, grad_eloc + jnp.multiply(eloc - E_mean, O) )
    #print("dominator:", dominator.shape)
    dominator = dominator.mean(axis = -1).real
    numerator = sign.mean().real

    gradient = dominator / numerator

    end = time.time()
    print("time for computing loss and grad:", end - start)

    return E_mean.real, gradient 


make_En_and_grad_vmapped = jax.vmap(make_En_and_grad, in_axes = (None, 0, 0, None, None, None, None, None), out_axes = (0, 0))



# sample sign, eloc, W, d_eloc/d_t, d_W/d_t 
@partial(jax.jit, static_argnums = (0, 4, 7, 8))
def sample_fori_loop(model, sigma, psi0, key_flip, batch, params, nthermal, nsample, ninterval):
    #nlayer = int(taus.shape[-1]/2)
    make_W, make_Eloc, make_eloc = init_fn(model)

    make_W_vmapped = jax.vmap(make_W, in_axes = (None, 0, None), out_axes = (0, 0))
    make_Eloc_vmapped = jax.vmap(make_Eloc, in_axes = (None, 0, None), out_axes = (0, 0))
    #make_eloc_vmapped = jax.vmap(make_eloc, in_axes = (None, 0, None), out_axes = (0, 0))

    make_grad_W = jax.jacrev(make_W, argnums = -1)
    make_grad_Eloc = jax.jacrev(make_Eloc, argnums = -1)
    make_grad_eloc = jax.jacrev(make_eloc, argnums = -1)

    make_grad_W_vmapped = jax.vmap(make_grad_W, in_axes = (None, 0, None), out_axes = (0, 0))
    make_grad_Eloc_vmapped = jax.vmap(make_grad_Eloc, in_axes = (None, 0, None), out_axes = (0, 0))
    make_grad_eloc_vmapped = jax.vmap(make_grad_eloc, in_axes = (None, 0, None), out_axes = (0, 0))

    ## thermal
    start = time.time()

    W_real, W_imag = make_W_vmapped(psi0, sigma, params)
    W = W_real + I * W_imag
    W_norm = abs(W)

    def mcmc_step(ithermal, state):
        key_flip, sigma, W, W_norm = state

        key_uniform, key_proposal, key_flip = jax.random.split(key_flip, 3)
        key_proposal = jax.random.split(key_proposal, batch)

        sigma_proposal = flip_vmapped(sigma, key_proposal)

        W_proposal_real, W_proposal_imag = make_W_vmapped(psi0, sigma_proposal, params)
        W_proposal = W_proposal_real + I * W_proposal_imag
        W_proposal_norm = abs(W_proposal)

        W_ratio = W_proposal_norm / W_norm
        ratio = jnp.where(W_ratio < 1., W_ratio, 1.)
        proposal = jax.random.uniform(key_uniform, shape = (batch, ))

        accept = ratio > proposal
        sigma = jnp.where(accept[:, None, None], sigma_proposal, sigma)
        W = jnp.where(accept, W_proposal, W)
        W_norm = abs(W)

        return key_flip, sigma, W, W_norm

    key_flip, sigma, W, W_norm = jax.lax.fori_loop(0, nthermal, mcmc_step, (key_flip, sigma, W, W_norm))

    end = time.time()
    #print("time for thermal_fori_loop:", end - start)

    ## collect
    start = time.time()

    ntau = params.shape[-1]
    W_sampled = jnp.zeros((nsample, batch)).astype('complex')
    sign_sampled = jnp.zeros((nsample, batch)).astype('complex')
    eloc_sampled = jnp.zeros((nsample, batch)).astype('complex')
    grad_W_sampled = jnp.zeros((nsample, batch, ntau)).astype('complex')
    grad_eloc_sampled = jnp.zeros((nsample, batch, ntau)).astype('complex')

    def body_fun_sample(isample, state):
        key_flip, sigma, W, W_norm, \
        W_sampled, sign_sampled, eloc_sampled, grad_W_sampled, grad_eloc_sampled = state

        W_sign = W / W_norm
        W_sampled = W_sampled.at[isample, :].set(W)
        sign_sampled = sign_sampled.at[isample, :].set(W_sign)
 
        Eloc_real, Eloc_imag = make_Eloc_vmapped(psi0, sigma, params) 
        Eloc = Eloc_real + I * Eloc_imag

        eloc = Eloc / W

        grad_W_real, grad_W_imag = make_grad_W_vmapped(psi0, sigma, params)     
        grad_W = grad_W_real + I * grad_W_imag

        grad_eloc_real, grad_eloc_imag = make_grad_eloc_vmapped(psi0, sigma, params)     
        grad_eloc = grad_eloc_real + I * grad_eloc_imag

        eloc_sampled = eloc_sampled.at[isample, :].set(eloc)
        grad_W_sampled = grad_W_sampled.at[isample, :, :].set(grad_W)
        grad_eloc_sampled = grad_eloc_sampled.at[isample, :, :].set(grad_eloc)

        ## flip
        for i in range(ninterval):
            key_flip, sigma, W, W_norm = mcmc_step(1, (key_flip, sigma, W, W_norm))

        return  key_flip, sigma, W, W_norm, \
                W_sampled, sign_sampled, eloc_sampled, grad_W_sampled, grad_eloc_sampled

    key_flip, sigma, W, W_norm, W_sampled, sign_sampled, eloc_sampled, grad_W_sampled, grad_eloc_sampled \
    = jax.lax.fori_loop(0, nsample, body_fun_sample, (key_flip, sigma, W, W_norm, \
       W_sampled, sign_sampled, eloc_sampled, grad_W_sampled, grad_eloc_sampled))

    end = time.time()
    #print("time for sample_fori_loop:", end - start)

    W_sampled = jnp.array(W_sampled).flatten()
    sign_sampled = jnp.array(sign_sampled).flatten()
    eloc_sampled = jnp.array(eloc_sampled).flatten()

    grad_W_sampled = jnp.array(grad_W_sampled)
    grad_eloc_sampled = jnp.array(grad_eloc_sampled)

    grad_W_sampled = jnp.concatenate(grad_W_sampled, axis =0).T
    grad_eloc_sampled = jnp.concatenate(grad_eloc_sampled, axis =0).T

    return W_sampled, sign_sampled, eloc_sampled, grad_W_sampled, grad_eloc_sampled


#sample_fori_loop_vmapped = jax.vmap(sample_fori_loop, in_axes = (None, 0, 0, None, None, None, None, None), out_axes = (0, 0, 0))



def make_En_and_grad_new(model, sigma, psi0, key, batch, params, nthermal, nsample, ninterval):
    start = time.time()
    W, sign, eloc, grad_W, grad_eloc = sample_fori_loop(model, sigma, psi0, key, batch, params, nthermal, nsample, ninterval)
    end = time.time()
    print("time for sample_fori_loop:", end - start)

    start = time.time()

    sign_dot_eloc = jnp.multiply(sign, eloc)
    E_mean = sign_dot_eloc.sum().real / sign.sum().real

    O = grad_W / W

    dominator = jnp.multiply( sign, grad_eloc + jnp.multiply(eloc - E_mean, O) )
    dominator = dominator.mean(axis = -1).real
    numerator = sign.mean().real

    gradient= dominator / numerator
    #print("sign_mean:")
    #print(sign.mean())

    end = time.time()
    print("time for computing loss and grad:", end - start)

    return E_mean, gradient, sign.mean()


make_En_and_grad_new_vmapped = jax.vmap(make_En_and_grad_new, in_axes = (None, None, 0, 0, None, None, None, None, None), \
                                       out_axes = (0, 0, 0))

make_En_and_grad_new_pmapped = jax.pmap(make_En_and_grad_new_vmapped, in_axes = (None, None, 0, None, None, None, None, None, None), \
                                       out_axes = (0, 0, 0), static_broadcasted_argnums = (0, 4, 7, 8))


# =========================================================================================================

def optimize_En(model, psi0, batch, nthermal, nsample, ninterval, Nlayer):
    
    #taus = jnp.array([0.1] * 2 * Nlayer) 
    taus = jnp.array([0.1, 0.1])
    params = taus

    learning_rate = 1e-2 * 2

    import optax
    optimizer = optax.adam(learning_rate = learning_rate)
    opt_state = optimizer.init(params)

    key = jax.random.PRNGKey(42)

    def step(params, opt_state, key):
        key_old, key = jax.random.split(key, 2)
        loss, grad = make_En_and_grad(model, psi0, key, batch, params, nthermal, nsample, ninterval)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, grad, key

    opt_nstep = 200

    model_ED = Hubbard_2d_ED(model.Lsite, model.N, model.t, model.U)
    eigvals_ED, _ = model_ED.eigs()
    E_exact = eigvals_ED[0]

    loss_all = []
    for istep in range(opt_nstep):
        start = time.time()
        params, opt_state, loss, grad, key = step(params, opt_state, key)
        end = time.time()
        print("time:", end - start)
        print('istep:', istep)
        print('grad:')
        print(grad)
        print('params:')
        print(params)
        print('loss, exact:', loss, E_exact)
        loss_all.append(loss)
        #if abs(loss - E_exact[0]) < 1e-2:
        #    break;
        print('\n')


# ==========================================================================

def make_p(qq):
    pp = jax.nn.softmax(qq)
    return pp

def make_logp(qq):
    pp = jax.nn.softmax(qq)
    logpp = jnp.log(pp)
    return logpp

make_grad_logp = jax.jacrev(make_logp, argnums = 0)


def make_loss_grad(beta, model, sigma, psi0_set, key, batch, params, nthermal, nsample, ninterval):
    # psi0_set is a 4d array of shape (nGPU, npsi0/nGPU, model.Lsite*2, model.N)
    npsi0 = psi0_set.shape[0] * psi0_set.shape[1]
    qq = params[:npsi0]
    pp = jax.nn.softmax(qq)
    grad_log_pp = make_grad_logp(qq)

    S_all = 1./beta * jnp.log(pp) 

    key_set = jax.random.split(key, psi0_set.shape[1])
    taus = params[npsi0:]

    E_all, grad_E_all, sign_mean = make_En_and_grad_new_pmapped(model, sigma, psi0_set, key_set, batch, taus, nthermal, nsample, ninterval)
    # E_all has shape of (nGPU, npsi0/nGPU)
    # grad_E_all has shape of (nGPU, npsi0/nGPU, ntau)
    # sign_mean has shape of (nGPU, nspi0/nGPU)

    E_all = jnp.concatenate(E_all, axis = 0)
    grad_E_all = jnp.concatenate(grad_E_all, axis = 0)
    sign_mean = jnp.concatenate(sign_mean, axis = 0)

    F = jnp.dot(pp, S_all + E_all)

    grad_F_qq = jnp.dot(jnp.multiply(S_all + E_all, grad_log_pp.T), pp) # shape of (qq.shape, )
    #print("grad_F_qq:", grad_F_qq.shape)
    grad_F_taus = jnp.dot(pp, grad_E_all)
    #print("grad_F_taus:", grad_F_taus.shape)

    grad_F = jnp.hstack((grad_F_qq, grad_F_taus))
    
    return F, grad_F, sign_mean.mean().real


def make_free_energy_ED(beta, Lx, Ly, N, t, U):
    model = Hubbard_2d_ED(Lx, Ly, N, t, U)
    F = model.free_energy(beta)
    return F


def optimize_F(beta, model, psi0_set, batch, nthermal, nsample, ninterval, Nlayer):
    # psi0_set is a 4d array of shape (nGPU, npsi0/nGPU, model.Lsite*2, model.N)
    npsi0 = psi0_set.shape[0] * psi0_set.shape[1]
    qq = jnp.array([0.1] * npsi0)    

    taus = jnp.array([0.05] * 2 * Nlayer) 
    params = jnp.hstack((qq, taus))

    key_init = jax.random.PRNGKey(21)
    sigma_init = random_init_sigma_vmapped(batch, Nlayer, model.Lsite*2, key_init)  # shape: (batch, nlayer, model.Lsite*2)

    learning_rate = 1e-2 

    import optax
    optimizer = optax.adam(learning_rate = learning_rate)
    opt_state = optimizer.init(params)

    key = jax.random.PRNGKey(42)

    def step(params, opt_state, key):
        key_old, key = jax.random.split(key, 2)
        loss, grad, sign_mean = make_loss_grad(beta, model, sigma_init, psi0_set, key, batch, params, nthermal, nsample, ninterval)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, grad, sign_mean, key

    opt_nstep = 2000

    F_exact = make_free_energy_ED(beta, model.Lx, model.Ly, model.N, model.t, model.U)

    loss_all = []
    sign_mean_all = []
    #params_final = []

    for istep in range(opt_nstep):
        start = time.time()
        params, opt_state, loss, grad, sign_mean, key = step(params, opt_state, key)
        end = time.time()
        print('istep:', istep)
        print('time for step:', end - start)
        #print('grad:')
        #print(grad)
        #print('params:')
        #print(params)
        print('taus:')
        print(params[psi0_set.shape[0]:])  
        print('loss, exact:', loss, F_exact)
        #print('loss:', loss)
        print('sign_mean:', sign_mean)
        loss_all.append(loss)
        sign_mean_all.append(sign_mean)
        #if abs(loss - F_exact) < 1e-2 * 5:
        #    break
        print('\n')

    datas = {"U": model.U, "beta": beta, "F_exact": F_exact, \
             "learning_rate": learning_rate, "opt_nstep":opt_nstep, \
            "loss": loss_all, "sign_mean": sign_mean_all, \
            "nlayer": Nlayer, "params_final": params, "npsi": npsi0}

    import pickle as pk
    fp = open('./optimize_F.txt', 'wb')
    pk.dump(datas, fp)
    fp.close()



# test ===========================================================================================

def test_sample_simple():
    L = 2
    N = int(L/2)
    t = 1.
    U = 1.

    tau1, tau2 = 0.2, 0.2
    taus = jnp.array([tau1, tau2])

    nthermal = 10
    nsample = 5
    ninterval = 1
    batch = 3

    key = jax.random.PRNGKey(21)

    model = Hubbard_2d_free(L, N, t, U)
    psi0 = model.get_ground_state()

    sigma, W, sign = sample_simple(model, psi0, key, batch, taus, nthermal, nsample, ninterval)

    print("sigma:", sigma.shape)   
    print("W:", W.shape)
    print("sign:", sign.shape)



def make_direct(Lx, Ly, N, t, U, taus):
    model = Hubbard_2d_ED(Lx, Ly, N, t, U)
    
    Tmatr = model.get_T()
    Umatr = model.get_U()
    Hmatr = model.get_Hamiltonian()

    model_free = Hubbard_2d_ED(Lx, Ly, N, t, 0.)
    es, psi = model_free.diagonalize()
    #print("es:")
    #print(es)
    psi0 = psi[:, 0]
    #print(psi0)
    #print('psi0 norm:')
    #print(jnp.linalg.norm(psi0))

    lenth = taus.shape[-1]
    nlayer = int(lenth/2)

    for ilayer in range(nlayer):
        tau1 = taus[lenth - ilayer*2 - 1]
        tau2 = taus[lenth - ilayer*2 - 2]
        expT = jax.scipy.linalg.expm(-I * Tmatr * tau2)
        expU = jax.scipy.linalg.expm(-I * Umatr * tau1)
        psi0 = jnp.dot(expT, jnp.dot(expU, psi0))

    E = jnp.dot(jnp.conjugate(psi0.T), jnp.dot(Hmatr, psi0))
    E = E / jnp.dot(jnp.conjugate(psi0.T), psi0)
    return E


def test_make_En_and_grad():
    Lx, Ly = 2, 2
    Lsite = Lx * Ly
    N = int(Lsite/2)
    t = 1.
    U = 0.

    model = Hubbard_2d_free(Lx, Ly, N, t, U)
    psi0 = model.get_ground_state()
    print("psi0:")
    print(psi0)

    nthermal = 50
    nsample = 2
    ninterval = 1
  
    batch = 500

    key = jax.random.PRNGKey(42)
    taus = jnp.array([0.3, 0.3])
    Nlayer = 1

    key_init = jax.random.PRNGKey(21)
    sigma = random_init_sigma_vmapped(batch, Nlayer, model.Lsite*2, key_init)

    En, _, _ = make_En_and_grad_new(model, sigma, psi0, key, batch, taus, nthermal, nsample, ninterval)
    print("En_MC:", En)

    En_ED = make_direct(Lx, Ly, N, t, U, taus)
    print("En_ED:", En_ED)

 
def test_optimize_En():
    L = 2
    N = int(L/2)
    t = 1.
    U = 3.
    model = Hubbard_2d_free(L, N, t, U)
    psi0 = model.get_ground_state()

    Nlayer = 1
 
    nthermal = 100
    nsample = 6
    ninterval = 5
  
    batch = 500

    optimize_En(model, psi0, batch, nthermal, nsample, ninterval, Nlayer)


def test_optimize_F():
    Lx, Ly = 3, 2
    Lsite = Lx * Ly
    N = int(Lsite/2)
    t = 1.
    U = 1.
    model = Hubbard_2d_free(Lx, Ly, N, t, U)

    #print(model.get_Hfree_half())
    #exit()

    psi0_set = model.get_psi0_full()
    #npsi0 = 100
    #psi0_set = model.get_psi0_nset(npsi0)
    psi0_set = jnp.array(psi0_set)
    print(psi0_set.shape[0])
    npsi0 = psi0_set.shape[0]
    nGPU = 8
    assert npsi0 % nGPU == 0
    psi0_set = jnp.split(psi0_set, nGPU)
    psi0_set = jnp.array(psi0_set)
    print('psi0_set.shape:', psi0_set.shape)

    nthermal = 50
    nsample = 5
    ninterval = 1
    batch = 500

    nlayer = 1
    beta = 1.

    optimize_F(beta, model, psi0_set, batch, nthermal, nsample, ninterval, nlayer)

 
# run ==================================================================================

#test_sample_simple()
#test_sample_vmapped()
#test_make_En_and_grad()
#test_optimize_En()
test_optimize_F()

