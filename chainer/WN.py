
import numpy as np
from chainer import functions as F
from chainer import links as L


class WN(L.Linear):

    def __init__(self, *args, **kwargs):
        super(WN, self).__init__(*args, **kwargs)
        self.add_param('g', self.W.data.shape[0])
        norm = np.linalg.norm(self.W.data, axis=1)
        self.g.data[...] = norm

    def __call__(self):
        """Applies the linear layer.
        Args:
            x (~chainer.Variable): Batch of input vectors.
        Returns:
            ~chainer.Variable: Output of the linear layer.
        """
        norm = F.batch_l2_norm_squared(self.W) ** 0.5
        norm_broadcasted = F.broadcast_to(
            F.expand_dims(norm, 1), self.W.data.shape)
        g_broadcasted = F.broadcast_to(
            F.expand_dims(self.g, 1), self.W.data.shape)
        return g_broadcasted * self.W / norm_broadcasted
