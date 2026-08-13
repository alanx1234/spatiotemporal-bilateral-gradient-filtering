import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import factorized
from scipy.stats.qmc import Sobol


def build_laplacian(faces, num_vertices):
    '''
    laplacian mesh version 
    (L @x)[i] measures how far vertex i sits from the avg of its neighbors

    faces: (f, 3) vertex indices
    num_vertices: unique vertices
    '''
    # each face has 3 edges 
    edges = np.concatenate([faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]])
    i,j = edges[:,0], edges[:,1]

    # 1 when two vertices share an edge
    A = sparse.coo_matrix((np.concatenate([np.ones(len(i)), np.ones(len(i))]), (np.concatenate([i, j]), np.concatenate([j, i]))),
        shape=(num_vertices, num_vertices))
    
    A = (A > 0).astype(float) # interior edges are double counted 
    deg = np.asarray(A.sum(axis=1)).ravel() # num neighbors
    L = sparse.diags(deg) - A 

    return L.tocsc()
    

class laplacian_smoothing_filter:
    '''
    laplacian smoothing filter for the spatial filtering step
    of large steps (section 5.2.1, for prefilter and postfilter)

    the idea is that the Laplacian groups each vertex to its edge-neighbors
    so that they move together
    '''
    def __init__(self, faces, num_vertices, strength):
        '''
        faces: (f, 3) triangle indices
        num_vertices: unique vertices
        strength: smoothing strength
        '''
        self.faces = faces
        self.num_vertices = num_vertices
        self.strength = strength

        L = build_laplacian(faces, num_vertices)                    
        eq = sparse.identity(num_vertices, format='csc') + strength * L   # I + λL
        self.solve = factorized(eq) # prepare to solve for u

    def __call__(self, values, guide=None):
        '''
        solve (I + lambda L) x = values

        values: (n, 3) signal on the vertices 
        guide: unused
        '''
        return self.solve(values)


class cross_bilateral_filter:
    '''
    use the Sobol sequence to sample directions on the unit sphere
    to convert from the square to the sphere

    read each vertex's normal and smooth vertices facing teh same way 
    for example: two vertices on the same flat cube face are smoothed together,
    but two vertices on opposite sides of a cube edge are not smoothed together 
    
    '''

    def __init__(self, faces, num_vertices, strength, sigma_data, sigma_phi, num_samples=32):
        '''
        faces: (f, 3) triangle indices
        num_vertices: number of unique vertices
        strength: lambda for the laplacian step
        sigma_data: von mises-fisher sharpness for data term
        num_samples: num of directions sampled on the sphere
        '''
        # 32 directions on unit sphere with Sobol sequence, representing
        # all orientations a surface could face
        # d=2 gives evenly spread pairs in the unit square, which we want to
        # bend onto a sphere
        sampler = Sobol(d=2, scramble=True, seed =0)

        eta = sampler.random(num_samples) 
        theta = 2 * np.pi * eta[:, 0]
        phi = np.arccos(1-2 * eta[:, 1])

        self.samples = np.stack([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),], axis=1)

        self.sigma_data = sigma_data
        self.sigma_phi = sigma_phi
        self.laplacian = laplacian_smoothing_filter(faces, num_vertices, strength)

    def __call__(self, values, normals):
        '''
        apply the cross-bilateral filter to values
        using the vertex normals as a guide

        implementing equation 5

        the idea is that you have 32 bins (directions), and you smooth gradients (or momenta)
        in the same bins but not across bins so that you don't smooth across edges

        values: (n, k) signal on the vertices
        normals: (n, 3) vertex normals (guide)
        '''

        # align each vertex normal with the 32 sampled directions
        W = np.exp(normals @ self.samples.T / self.sigma_data) # (n,32)

        # stack the signal, weighed by direction, and also the weights
        num_channels = values[:, :, None] * W[:, None, :] # (n, 3, 32)
        stacked = np.concatenate([num_channels, W[:, None, :]], axis=1)  # (n, 4, 32)

        # use laplacian to smooth over the mesh
        diff = self.laplacian.solve(stacked.reshape(len(values), -1))
        diff = diff.reshape(len(values), values.shape[1] + 1, -1) # (n,4,32)

        # each vertex collects results from directions it points towards
        phi = np.exp(normals @ self.samples.T / self.sigma_phi) # (n, 32)
        weighted = diff *phi[:, None, :] # (n,4,32)

        # sum over 32 directions
        num = weighted[:,:values.shape[1],:].sum(axis=-1)   # (n, 3)
        denom = weighted[:,values.shape[1]:,:].sum(axis=-1) # (n, 1)

        return num/denom