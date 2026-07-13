import numpy as np


class laplacian_smoothing_filter:
    '''
    smooth 1d signal with laplacian smoothing, where you smooth values isotropically
    including across edges
    '''

    def __init__(self, strength):
        '''
        strength: how strongly to smooth input values (lambda)
        '''

        self.strength = strength

    def __call__(self, values, guide=None):
        '''
        apply laplacian smoothing to values

        values: 1d signal to smooth
        guide: unused, allows the same filter interface as cross bilateral filtering
        '''
        values = np.asarray(values, dtype=float)
        size = len(values)

        # 1d laplacian
        laplacian = (2 * np.eye(size) - np.eye(size, k=1)- np.eye(size, k=-1))

        # endpoints have one neighbor since there are no samples beyond
        laplacian[0, 0] = 1
        laplacian[-1, -1] = 1

        # (I + lambda * L) * smoothed values = values, solve for smoothed values
        return np.linalg.solve(np.eye(size) + self.strength * laplacian, values)

class cross_bilateral_filter:
    '''
    smooth 1d signal using edge-aware spatial smoothing, using the guide signal to avoid smoothing across edges
    '''
    def __init__(self, sigma_spatial, sigma_data, radius=None):
        '''
        sigma_spatial: smoothing spatial range
        sigma_data: threshold for difference btwn theta values before stopping
        radius: max number of neighbors on each side
        '''
        self.sigma_spatial = sigma_spatial
        self.sigma_data = sigma_data
        self.radius = (radius if radius is not None  
                       else int(np.ceil(3 * sigma_spatial)))

    def __call__(self, h, guide):
        '''
        apply a cross bilateral filter to h, using the guide (theta) for edge-aware weighting
        here, it is a discrete weighted average in the 1d case

        h: thing being filtered, here it is m_{t} or v_{t} of Adam
        guide: theta, the current parameter to guide the edge-aware
        ''' 

        filtered = np.zeros_like(h, dtype=float)

        for i in range(len(h)):
            start = max(0, i - self.radius)
            end = min(len(h), i + self.radius + 1)
            neighbors = np.arange(start, end) # simply for indices of neighbors

            # eq 4: spatial weights, in 1d this is simply based on distance btwn indices
            w_s = np.exp(-(abs(neighbors - i))/ self.sigma_spatial)
            # eq 4: data weights 
            w_d = np.exp(-(abs(guide[neighbors] - guide[i]))/self.sigma_data)
            # multiply the two weights together
            weights = w_s * w_d
            # weighted average of neighbors
            filtered[i] = np.sum(h[neighbors] * weights) / np.sum(weights)

        
        return filtered

