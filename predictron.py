import inspect

import chainer
from chainer import functions as F
from chainer import links as L


class Sequence(chainer.ChainList):

    def __init__(self, *layers):
        self.layers = layers
        links = [layer for layer in layers if isinstance(layer, chainer.Link)]
        super().__init__(*links)

    def __call__(self, x, test):
        h = x
        for layer in self.layers:
            argnames = inspect.getargspec(layer)[0]
            if 'test' in argnames:
                h = layer(h, test=test)
            else:
                h = layer(h)
        return h


class PredictronCore(chainer.Chain):

    def __init__(self, n_tasks, n_channels):
        super().__init__(
            state2hidden=Sequence(
                L.Convolution2D(n_channels, n_channels, ksize=3, pad=1),
                L.BatchNormalization(n_channels),
                F.relu,
            ),
            hidden2nextstate=Sequence(
                L.Convolution2D(n_channels, n_channels, ksize=3, pad=1),
                L.BatchNormalization(n_channels),
                F.relu,
                L.Convolution2D(n_channels, n_channels, ksize=3, pad=1),
                L.BatchNormalization(n_channels),
                F.relu,
            ),
            hidden2reward=Sequence(
                L.Linear(None, n_channels),
                L.BatchNormalization(n_channels),
                F.relu,
                L.Linear(n_channels, n_tasks),
            ),
            hidden2gamma=Sequence(
                L.Linear(None, n_channels),
                L.BatchNormalization(n_channels),
                F.relu,
                L.Linear(n_channels, n_tasks),
                F.sigmoid,
            ),
            hidden2lambda=Sequence(
                L.Linear(None, n_channels),
                L.BatchNormalization(n_channels),
                F.relu,
                L.Linear(n_channels, n_tasks),
                F.sigmoid,
            ),
        )

    def __call__(self, x, test):
        hidden = self.state2hidden(x, test=test)
        # No skip
        nextstate = self.hidden2nextstate(hidden, test=test)
        reward = self.hidden2reward(hidden, test=test)
        gamma = self.hidden2gamma(hidden, test=test)
        # lambda doesn't backprop errors to states
        lmbda = self.hidden2lambda(
            chainer.Variable(hidden.data), test=test)
        return nextstate, reward, gamma, lmbda


class Predictron(chainer.Chain):

    def __init__(self, n_tasks, n_channels, model_steps,
                 use_reward_gamma=True, use_lambda=True, usage_weighting=True):
        self.model_steps = model_steps
        self.use_reward_gamma = use_reward_gamma
        self.use_lambda = use_lambda
        self.usage_weighting = usage_weighting
        super().__init__(
            obs2state=Sequence(
                L.Convolution2D(None, n_channels, ksize=3, pad=1),
                L.BatchNormalization(n_channels),
                F.relu,
                L.Convolution2D(n_channels, n_channels, ksize=3, pad=1),
                L.BatchNormalization(n_channels),
                F.relu,
            ),
            core=PredictronCore(n_tasks=n_tasks, n_channels=n_channels),
            state2value=Sequence(
                L.Linear(None, n_channels),
                L.BatchNormalization(n_channels),
                F.relu,
                L.Linear(n_channels, n_tasks),
            ),
        )

    def unroll(self, x, test):
        # Compute g^k and lambda^k for k=0,...,K
        g_k = []
        lambda_k = []
        state = self.obs2state(x, test=test)
        g_k.append(self.state2value(state, test=test))  # g^0 = v^0
        reward_sum = 0
        gamma_prod = 1
        for k in range(self.model_steps):
            state, reward, gamma, lmbda = self.core(state, test=test)
            if not self.use_reward_gamma:
                reward = 0
                gamma = 1
            if not self.use_lambda:
                lmbda = 1
            lambda_k.append(lmbda)  # lambda^k
            v = self.state2value(state, test=test)
            reward_sum += gamma_prod * reward
            gamma_prod *= gamma
            g_k.append(reward_sum + gamma_prod * v)  # g^{k+1}
        lambda_k.append(0)  # lambda^K = 0
        # Compute g^lambda
        lambda_prod = 1
        g_lambda = 0
        w_k = []
        for k in range(self.model_steps + 1):
            w = (1 - lambda_k[k]) * lambda_prod
            w_k.append(w)
            lambda_prod *= lambda_k[k]
            # g^lambda doesn't backprop errors to g^k
            g_lambda += w * chainer.Variable(g_k[k].data)
        return g_k, g_lambda, w_k

    def supervised_loss(self, x, t):
        g_k, g_lambda, w_k = self.unroll(x, test=False)
        if self.usage_weighting:
            g_k_loss = sum(F.sum(w * (g - t) ** 2) / x.shape[0]
                           for g, w in zip(g_k, w_k))
        else:
            g_k_loss = sum(F.mean_squared_error(g, t) for g in g_k) / len(g_k)
        g_lambda_loss = F.mean_squared_error(g_lambda, t)
        return g_k_loss, g_lambda_loss

    def unsupervised_loss(self, x):
        g_k, g_lambda, w_k = self.unroll(x, test=False)
        # Only update g_k
        g_lambda.creator = None
        if self.usage_weighting:
            return sum(F.sum(w * (g - g_lambda) ** 2) / x.shape[0]
                       for g, w in zip(g_k, w_k))
        else:
            return sum(F.mean_squared_error(g, g_lambda) for g in g_k) / len(g_k)
