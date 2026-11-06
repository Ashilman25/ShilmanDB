#The embedding proxy: an L2 logistic regression fit by Newton's method
#using numpy only

import numpy as np

LAM = 1e-2
STEPS = 25
INTERCEPT_RIDGE = 1e-8


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


class Proxy:
    def fit(self, X, y, n_classes: int = 2) -> "Proxy":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        n = len(X)
        
        self.n_classes = n_classes
        self._mean = X.mean(axis=0)
        std = X.std(axis=0)
        self._std = np.where(std > 1e-12, std, 1.0)
        
        Xs = (X - self._mean) / self._std
        _, _, vt = np.linalg.svd(Xs, full_matrices=False)  # rows of vt span the training rows
        Z = np.hstack([Xs @ vt.T, np.ones((n, 1))])  # last column is the intercept
        r = Z.shape[1] - 1
        
        penalty = np.full(r + 1, LAM)
        penalty[r] = INTERCEPT_RIDGE

        if n_classes == 2:
            theta = self._newton_binary(Z, (y == 1).astype(float), penalty)
            coef = theta[:r]
            self._w = vt.T @ coef
            self._b = float(theta[r])
        else:
            theta = self._newton_softmax(Z, y, n_classes, penalty)
            self._w = vt.T @ theta[:r]
            self._b = theta[r]
            
        return self

    @staticmethod
    def _newton_binary(Z, t, penalty):
        n = len(Z)
        theta = np.zeros(Z.shape[1])
        pen = np.diag(penalty)
        
        for _ in range(STEPS):
            p = _sigmoid(Z @ theta)
            grad = Z.T @ (p - t) / n + penalty * theta * np.r_[np.ones(len(theta) - 1), 0.0]
            hess = (Z * (p * (1 - p))[:, None]).T @ Z / n + pen
            theta = theta - np.linalg.solve(hess, grad)
        return theta

    @staticmethod
    def _newton_softmax(Z, y, k, penalty):
        n, d = Z.shape
        Y = np.zeros((n, k))
        Y[np.arange(n), y] = 1.0
        W = np.zeros((d, k))
        
        pen_mask = np.r_[np.ones(d - 1), 0.0][:, None]
        for _ in range(STEPS):
            P = _softmax(Z @ W)
            grad = Z.T @ (P - Y) / n + penalty[:, None] * W * pen_mask
            hess = np.zeros((k * d, k * d))
            
            for a in range(k):
                for b in range(a, k):
                    s = (P[:, a] * (a == b) - P[:, a] * P[:, b]) / n
                    block = (Z * s[:, None]).T @ Z
                    hess[a * d:(a + 1) * d, b * d:(b + 1) * d] = block
                    if a != b:
                        hess[b * d:(b + 1) * d, a * d:(a + 1) * d] = block.T
                        
            hess[np.arange(k * d), np.arange(k * d)] += np.tile(penalty, k)
            step = np.linalg.solve(hess, grad.T.reshape(-1)).reshape(k, d).T
            W = W - step
        return W

    #Binary: p(yes), shape (n,)
    # more classes: probabilities, shape (n, k)
    def scores(self, X) -> np.ndarray:
        Xs = (np.asarray(X, dtype=np.float64) - self._mean) / self._std
        z = Xs @ self._w + self._b
        return _sigmoid(z) if self.n_classes == 2 else _softmax(z)
