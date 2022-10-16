import numpy as np
import jax
import jax.numpy as jnp

def test_outer():
    a = jnp.array([1, 1, 1, 1])
    b = jnp.array([2, 0, 3, 4])
    c = jnp.outer(a, b)
    print(c)

    arr = jnp.tile(a, (2, 1))
    brr = jnp.tile(b, (2, 1))
    crr = jnp.outer(arr, brr)
    print(crr)


def make_outer(arr):
    return jnp.outer(arr, jnp.conjugate(arr))

#make_outer_vmapped = jax.vmap(make_outer, in_axes = 0, out_axes = 0)
make_outer_vmapped = jax.vmap(jnp.outer, in_axes = (0, 0), out_axes = 0)


def test_make_outer():
    a = jnp.array([1, 1, 1, 1])
    b = jnp.array([2, 0, 3, 4])
    c = jnp.outer(a, b)
    print('c:')
    print(c)
    d = jnp.outer(a, b)
    print('d:')
    print(d)

    arr = jnp.tile(a, (2, 1))
    brr = jnp.tile(b, (2, 1))
 
    crr = make_outer_vmapped(arr, brr)
    print('crr:')
    print(crr) 


make_matmul_vmapped = jax.vmap(jnp.matmul, in_axes = (None, 0), out_axes = 0)

make_trace_vmapped = jax.vmap(jnp.trace, in_axes = 0, out_axes = 0)


def test_make_matmul():
    arr = jnp.array([[1, 0], [0, 1]])
    brr = jnp.array([ [[1, 0], [0, 1]], [[2, 0], [0, 2]] ])
    crr = make_matmul_vmapped(arr, brr)
    print(crr)
    tr = make_trace_vmapped(crr)
    print(tr)

    

# run ===============================================

#test_outer()
#test_make_outer()
test_make_matmul()
