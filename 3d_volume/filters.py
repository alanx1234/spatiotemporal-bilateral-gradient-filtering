import numpy as np

class laplacian_smoothing_filter:
    '''
    smooth 3d signal with laplacian smoothing, where you smooth values isotropically
    including across edges
    '''

    def __init__(self, passes=3, sigma_data=1e3):
        '''
        passes: number of a-trous passes (F), aka smoothing strength analog
        sigma_data: large enough that the data term never stops any smoothing
        '''

        # same idea as cross bilateral, just with the data term off (w_d)
        # sigma_data this big makes every w_d 1, so only w_s is left, which is just
        # laplacian smoothing (section 4.3)
        self.filter = cross_bilateral_filter(sigma_data, passes)

    def __call__(self, values, guide=None):
        '''
        apply laplacian smoothing to values

        values: 3d signal to smooth
        guide: unused, allows the same filter interface as cross bilateral filtering
        '''

        # w_d is 1 everywhere so the guide does nothing, but the filter still wants one
        return self.filter(values, values)


class cross_bilateral_filter:
    '''
    vectorized version of the cross-bilateral filter
    '''

    def __init__(self, sigma_data, passes=5):
        '''
        sigma_data: threshold for difference btwn theta values before stopping
        passes: number of a-trous passes (F), for spatial support
        '''

        self.sigma_data = sigma_data
        self.passes = passes

    def __call__(self, h, guide):
        '''
        apply a cross bilateral filter to h, using the guide (theta) for edge-aware weighting

        h: thing being filtered, here it is m_{t} or v_{t} of Adam
        guide: theta, the current parameter to guide the edge-aware filtering
        '''

        h = np.asarray(h, dtype=float)
        guide = np.asarray(guide, dtype=float)

        # fallback for density (scalar)
        scalar = h.ndim == 3
        if scalar:
            h = h[..., None]
            guide = guide[..., None]

        depth, height, width = h.shape[:3]
        kernel = [3.0/8, 1.0/4, 1.0/16]

        for i in range(self.passes):
            stride = 2 ** i
            # furthest sample is 2 strides out so pad that much on every side
            pad = 2 * stride
            
            # depth, height, width
            # (0, 0) on the last axis, channels aren't spatial so they get no neighbours
            widths = ((pad, pad), (pad, pad), (pad, pad), (0, 0))
            h_padded = np.pad(h, widths)
            guide_padded = np.pad(guide, widths)

            # 1 where the grid is and 0 for padding
            valid_padded = np.pad(np.ones(h.shape[:3] + (1,)), widths)


            num = np.zeros_like(h)
            denom = np.zeros_like(h)

            for dz in range(-2, 3, 1):
                for dy in range(-2, 3, 1):
                    for dx in range(-2, 3, 1):
                        rows = slice(pad + dy*stride, pad + dy*stride + height)
                        cols = slice(pad + dx*stride, pad + dx*stride + width)
                        planes = slice(pad + dz*stride, pad + dz*stride + depth)

                        # how spatially close in 3d space are they? 
                        # uses 5x5x5 weights, factored along each axis
                        w_s = kernel[abs(dz)] * kernel[abs(dy)] * kernel[abs(dx)]

                        # how similar are the guide's values? 
                        # albedo: (r,g,b) -> if we treat RGB as 3d space, the norm is just the straight line distance between two
                        # 3d points, which we calculate as the euclidean distance
                        # density: collapses to the 2d version since it has 1 channel
                        difference = guide_padded[planes, rows, cols] - guide
                        w_d = np.exp(-np.linalg.norm(difference, axis=-1, keepdims=True) / self.sigma_data)

                        weights = w_s * w_d * valid_padded[planes, rows, cols]
                        num += weights * h_padded[planes, rows, cols] 
                        denom += weights

            # one divide for the whole grid instead of one per voxel
            h = num / denom

        return h[..., 0] if scalar else h



