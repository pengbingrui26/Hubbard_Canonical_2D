import numpy as np
import jax
import jax.numpy as jnp
import pickle as pk

fd = open('./optimize_F.txt', 'rb')
datas = pk.load(fd)
fd.close()

F_exact = datas['F_exact']
loss = datas['loss']
for il, ll in enumerate(loss):
   print(il, ll)

print(loss[-1]/F_exact)



