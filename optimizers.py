import numpy as np


class adam_optimizer:
    '''
    optimize signal with adam, which temporally filters gradients
    '''

    def __init__(self, lr, b_1=0.9, b_2=0.999, eps=1e-8):
        '''
        lr: learning rate
        b_1: strength of m_t
        b_2: temporal smoothing strength for the second moment, (v_t)
        eps: avoid div by zero
        '''
        self.lr = lr
        self.b_1 = b_1
        self.b_2 = b_2
        self.eps = eps
        self.m = None
        self.v = None
        self.iteration = 0

    def step(self, theta, grad):
        '''
        apply one adam update to theta using the current gradient

        theta: current parameter values
        grad: gradient of loss wrt theta
        '''
        self.iteration += 1

        if self.m is None:
            self.m = np.zeros_like(theta, dtype=float)
            self.v = np.zeros_like(theta, dtype=float)

        m_t = self.b_1 * self.m + (1 - self.b_1) * grad 
        v_t = self.b_2 * self.v + (1 - self.b_2) * (grad ** 2)

        self.m = m_t
        self.v = v_t

        #bias correction
        m_hat = self.m / (1 - self.b_1 ** self.iteration)
        v_hat = self.v / (1 - self.b_2 ** self.iteration)

        #theta_{t+1}
        return theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class spatiotemporal_optimizer:
    '''
    section 4.1 in the paper
    optimize signal with adam and cross-bilateral spatial filtering
    '''
    def __init__(
        self,
        lr,
        filter,
        b_1=0.9,
        b_2=0.999,
        eps=1e-8,
    ):
        '''
        lr: learning rate
        filter: cross-bilateral filter for m_t and v_t ( from filters.py )
        beta_1: strength of m_t
        beta_2: strength of v_t
        epsilon: avoid div by zero
        '''
        self.lr = lr
        self.filter = filter
        self.b_1 = b_1
        self.b_2 = b_2
        self.eps = eps
        self.m = None
        self.v = None
        self.iteration = 0

    def step(self, theta, grad):
        '''
        apply one spatiotemporal update to theta with current gradient

        theta: current parameter values for spatial filtering (acts as the guide)
        grad: gradient of loss wrt theta
        '''
        if self.m is None:
            self.m = np.zeros_like(theta, dtype=float)
            self.v = np.zeros_like(theta, dtype=float)
            
        self.iteration += 1

        # store initial adam m, v
        self.m = self.b_1 * self.m + (1 -self.b_1) * grad
        self.v = self.b_2 * self.v + (1 - self.b_2) * (grad ** 2)

        # postfilter using theta as guide
        m_tilde = self.filter(self.m, theta)
        v_tilde = self.filter(self.v, theta)

        #bias correction after postfiltering
        m_hat = m_tilde/ (1 - self.b_1**self.iteration)
        v_hat = v_tilde/ (1 - self.b_2**self.iteration)

        # theta _{t+1}
        return theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class spatiotemporal_mesh_optimizer:
    '''
    section 5.2 in the paper (large steps update)

    this is the mesh version of the spatiotemporal optimizer.
      - prefilters the raw gradient with laplacian smoothing
      - adam's second moment uses the infinity norm ||v||_inf (a single scalar) 
      - postfilter is guided by the vertex normals
    '''

    def __init__(
        self,
        lr,
        prefilter,
        postfilter,
        b_1=0.9,
        b_2=0.999,
        eps=1e-8,
    ):
        '''
        lr: learning rate
        prefilter: filter applied to the raw gradient (laplacian)
        postfilter: filter applied to m_t that uses normals as guide (laplacian or cross-bilateral)
        b_1: strength of m_t
        b_2: strength of v_t
        eps: avoid div by zero
        '''
        self.lr = lr
        self.prefilter = prefilter
        self.postfilter = postfilter
        self.b_1 = b_1
        self.b_2 = b_2
        self.eps = eps
        self.m = None
        self.v = None
        self.iteration = 0

    def step(self, theta, grad, normals):
        '''
        apply one Large Steps update to the vertex positions

        theta: current vertex positions
        grad: gradient of loss wrt the vertices
        normals: vertex normals aka the guide
        '''

        self.iteration += 1
        if self.m is None:
            self.m = np.zeros_like(theta,dtype=float)
            self.v = np.zeros_like(theta,dtype=float)

        # 2: prefilter raw gradient
        grad = self.prefilter(grad)

        # 3-4: adam moments
        self.m = self.b_1 * self.m + (1 - self.b_1) * grad
        self.v = self.b_2 * self.v + (1 - self.b_2) * (grad ** 2)

        # 5: postfilter m, guided by normals
        m_tilde = self.postfilter(self.m, normals)
        # 6: uniformadam (one scalar for whole mesh)
        v_tilde = np.abs(self.v).max()

        # bias correction
        m_hat = m_tilde / (1 - self.b_1 ** self.iteration)
        v_hat = v_tilde / (1 - self.b_2 ** self.iteration)

        # 7
        return theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

