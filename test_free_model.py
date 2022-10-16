import numpy as np
import jax
import jax.numpy as jnp

from free_model import Hubbard_2d_free
from ED_2d import Hubbard_2d_ED

jax.config.update("jax_enable_x64", True)

def test_H_free():
    Lx, Ly, t, U = 8, 8, 1., 0., 1.
    N = int(Lx*Ly/2)
    model = Hubbard_2d_free(Lx, Ly, N, t, U)

    H_free_half = model.H_free_half
    print(H_free_half)
    
    idx = model.idx
    print(idx)
    exit()

def test_ground_energy():
    Lx, Ly, t, U = 4, 2, 1., 0.,
    N = int(Lx*Ly/2)
    model = Hubbard_2d_free(Lx, Ly, N, t, U)
    eg = model.get_ground_energy()
    print(eg)

    model_ED = Hubbard_2d_ED(Lx, Ly, N, t, 0.)
    eigvals, _ = model_ED.diagonalize()
    eg_ED = eigvals[0]
    print(eg_ED)


def test_get_psi0_nset():
    Lx, Ly, t, U = 4, 4, 1., 1.
    N = int(Lx*Ly/2)
    model = Hubbard_2d_free(Lx, Ly, N, t, U)
    npsi = 3
    psi0_ful_nset = model.get_psi0_nset(3)
    for psi0 in psi0_ful_nset:
        print(jnp.round(psi0, 2))


def draw_2D_Fermi_sea_4():
    import matplotlib.pyplot as plt

    xx = [ 0, \
          -jnp.pi/2, 0, jnp.pi/2, \
          0 ]
   
    yy = [ jnp.pi/2, \
          0, 0, 0, \
          -jnp.pi/2 ]

    xxx = [ -jnp.pi/2, jnp.pi/2, \
            -jnp.pi/2, jnp.pi/2 ]

    yyy = [ jnp.pi/2, jnp.pi/2, \
            -jnp.pi/2, -jnp.pi/2 ]

    plt.figure(figsize = (6, 6))

    plt.title('2D BZ, 5 blue dots, 4 orange dots')
    plt.scatter(xx, yy)
    plt.scatter(xxx, yyy)
    
    plt.xlim(-jnp.pi, jnp.pi)
    plt.ylim(-jnp.pi, jnp.pi)
    #plt.plot()
    plt.show()


def draw_2D_Fermi_sea_6():
    import matplotlib.pyplot as plt

    xx = [ 0, \
          -np.pi/3, 0, np.pi/3, \
          -np.pi*2/3, -np.pi/3, 0, np.pi/3, np.pi*2/3, \
          -np.pi/3, 0, np.pi/3, \
          0 ]
   
    yy = [ np.pi/3*2, \
          np.pi/3, np.pi/3, np.pi/3, \
          0, 0, 0, 0, 0, \
          -np.pi/3, -np.pi/3, -np.pi/3, \
          -np.pi/3*2 ]

    xxx = [ -np.pi/3, np.pi/3, \
            -np.pi/3*2, np.pi/3*2, \
            -np.pi/3*2, np.pi/3*2, \
            -np.pi/3, np.pi/3 ]

    yyy = [ np.pi/3*2, np.pi/3*2, \
            np.pi/3, np.pi/3, \
            -np.pi/3, -np.pi/3, \
            -np.pi/3*2, -np.pi/3*2 ]

    plt.figure(figsize = (6, 6))

    plt.title('2D BZ, 13 blue dots, 8 orange dots')
    plt.scatter(xx, yy)
    plt.scatter(xxx, yyy)
    
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)
    #plt.plot()
    plt.show()



# run =================================================================

#test_H_free()
#test_ground_energy()
test_get_psi0_nset()

