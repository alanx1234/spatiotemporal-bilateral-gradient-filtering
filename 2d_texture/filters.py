import numpy as np

class laplacian_smoothing_filter:
    '''
    smooth 2d signal with laplacian smoothing, where you smooth values isotropically
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

        values: 2d signal to smooth
        guide: unused, allows the same filter interface as cross bilateral filtering
        '''

        # w_d is 1 everywhere so the guide does nothing, but the filter still wants one
        return self.filter(values, values)

class cross_bilateral_filter_nonvectorized:
    '''
    smooth 2d signal using edge-aware spatial smoothing, using guide signal to avoid smoothing across edges

    to use the standard bilateral filter, we pass in filter(m_t, guide=m_t) for instance

    implemented using a-trous passes (section 5.1)

    this is the nonvectorized version which is a lot more inefficient, but was the first attempt at reimplementing i did
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
        guide: theta, the current parameter to guide the edge-aware

        conceptually, the idea here is that we use a 5x5 table with grid weights to calculate w_s to determine
        how spatially close two texels are in a 2d grid
        '''
        h = np.asarray(h, dtype=float)
        guide = np.asarray(guide, dtype=float)

        height, width = h.shape
        kernel = [3.0/8, 1.0/4, 1.0/16]

        for i in range(self.passes):
            filtered = np.zeros_like(h)
            # each pass reads neighbors 1 texel away, 2, 4, ...
            stride = 2 ** i 
            
            # center the 5x5 table on the texel you're computing, multiply each covered texel by the weight on that cell
            for y in range(height):
                for x in range(width):
                    num = denom = 0.0
                    
                    # actual kernel [1/16, 1/4, 3/8, 1/4, 1/16] has 5 entries, offset of 2 on each side
                    for dy in range(-2, 3, 1):
                        for dx in range(-2, 3, 1):
                            nx, ny = x + dx*stride, y + dy*stride 
                            
                            # skip out of range neighbors
                            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                                continue
                            
                            # how spatially close in 2d space are they?
                            w_s = kernel[abs(dy)] * kernel[abs(dx)]
                            # how similar are the guide's values? 
                            # for the standard bilateral filter, this is just m_t or v_t itself
                            # but for cross-bilateral, it is the texel value itself
                            w_d = np.exp(-abs(guide[ny, nx] - guide[y, x]) / self.sigma_data)

                            num += w_s * w_d * h[ny, nx]
                            denom += w_s * w_d 

                    filtered[y,x] = num / denom
            # feed this pass's output into the next one, so F passes reaches roughly ~2^F texels
            #e.g. pass 0, stride 1 reads 5 consecutive texels
            # pass 3, stride 8 reads 8 texels apart
            h = filtered

        return h


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

        height, width = h.shape
        kernel = [3.0/8, 1.0/4, 1.0/16]

        for i in range(self.passes):
            stride = 2 ** i
            # furthest sample is 2 strides out so pad that much on every side
            pad = 2 * stride
            h_padded = np.pad(h, pad) # values being averaged
            guide_padded = np.pad(guide, pad) # for computing w_d
            # 1 on real texels and 0 on the padding
            valid_padded = np.pad(np.ones_like(h), pad) # 1 where the grid is and 0 on the border

            #visualization:
            #dy=0, dx=0          dy=-1, dx=0         dy=0, dx=-1
            #rows 2:5            rows 1:4            cols 1:4

            #1 2 3               0 0 0               0 1 2
            #4 5 6               1 2 3               0 4 5
            #7 8 9               4 5 6               0 7 8

            num = np.zeros_like(h)
            denom = np.zeros_like(h)

            for dy in range(-2, 3, 1):
                for dx in range(-2, 3, 1):
                    # whole grid  is shifted by this offset
                    rows = slice(pad + dy*stride, pad + dy*stride + height)
                    cols = slice(pad + dx*stride, pad + dx*stride + width)

                    # how spatially close in 2d space are they? 
                    w_s = kernel[abs(dy)] * kernel[abs(dx)]
                    # how similar are the guide's values? 
                    # only the neighbor moves
                    w_d = np.exp(-np.abs(guide_padded[rows, cols] - guide) / self.sigma_data)

                    weights = w_s * w_d * valid_padded[rows, cols]
                    num += weights * h_padded[rows, cols]
                    denom += weights

            # one divide for the whole grid instead of one per texel
            h = num / denom

        return h


