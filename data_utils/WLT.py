import numpy as np


class WLT(object):
    def __init__(self, args):
        self.num_anchor = args.num_anchor
        self.sigma = 0.5
        self.R_alpha = args.R_alpha
        self.S_size = args.S_size
        self.seed = args.seed
        
    def __call__(self, pos):
        M = self.num_anchor 

        # Multi-anchor transformation
        idx = self.fps(pos, M)
        pos_anchor = pos[idx]
        pos_repeat = np.expand_dims(pos,0).repeat(M, axis=0)
        pos_normalize = np.zeros_like(pos_repeat, dtype=pos.dtype)
        pos_normalize = pos_repeat - pos_anchor.reshape(M,-1,3)
        pos_transformed = self.multi_anchor_transformation(pos_normalize)
        
        # Smooth Aggregation
        pos_transformed = pos_transformed + pos_anchor.reshape(M,-1,3)        
        pos_new = self.smooth_aggregation(pos, pos_anchor, pos_transformed)
        return pos.astype('float32'), pos_new.astype('float32')
        
    def fps(self, pos, npoint):
        np.random.seed(self.seed)
        N, _ = pos.shape
        centroids = np.zeros(npoint, dtype=np.int_)
        distance = np.ones(N, dtype=np.float64) * 1e10
        farthest = np.random.randint(0, N, (1,), dtype=np.int_)
        for i in range(npoint):
            centroids[i] = farthest
            centroid = pos[farthest, :]
            dist = ((pos - centroid)**2).sum(-1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = distance.argmax()
        return centroids
    
    def multi_anchor_transformation(self, pos_normalize):
        M, _, _ = pos_normalize.shape

        degree = np.pi * np.ones((M, 3)) * self.R_alpha / 180.0
        scale = np.ones((M, 3)) * self.S_size

        # Scaling Matrix
        S = np.expand_dims(scale, axis=1)*np.eye(3)

        # Rotation Matrix
        sin = np.sin(degree)
        cos = np.cos(degree)
        sx, sy, sz = sin[:,0], sin[:,1], sin[:,2]
        cx, cy, cz = cos[:,0], cos[:,1], cos[:,2]
        R = np.stack([cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx,
             sz*cy, sz*sy*sx + cz*cy, sz*sy*cx - cz*sx,
             -sy, cy*sx, cy*cx], axis=1).reshape(M,3,3)
        
        pos_normalize = pos_normalize @ R @ S
        return pos_normalize
    
    def smooth_aggregation(self, pos, pos_anchor, pos_transformed):
        M, N, _ = pos_transformed.shape
        
        # Distance between anchor points & entire points
        sub = np.expand_dims(pos_anchor,1).repeat(N, axis=1) - np.expand_dims(pos,0).repeat(M, axis=0)
        projection = np.expand_dims(np.eye(3), 0)
        sub = sub @ projection
        sub = np.sqrt(((sub) ** 2).sum(2))

        # Kernel regression
        weight = np.exp(-0.5 * (sub ** 2) / (self.sigma ** 2))
        pos_new = (np.expand_dims(weight,2).repeat(3, axis=-1) * pos_transformed).sum(0)
        pos_new = (pos_new / weight.sum(0, keepdims=True).T)
        return pos_new
