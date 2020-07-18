# this little script is useful for rendering a bunch of sample sets at once by iterating different values of r and/or k

import n_dims_poisson_sampling as p

for k in [10,20,30]:
    for n_dims in range(8,10):
        r = n_dims * 0.025
        p.makeFile(n_dims,r,k)