import torch.optim

"""
A reimplementation of the conjugate gradient algorithm (warts and all) used in
XDGRASP.

Original source here: https://cai2r.net/resources/software
"""
class CG(torch.optim.Optimizer):
    """
    Nonlinear conjugate gradient descent.
    The names of the variables were kept from the Matlab implementation to
    facilitate comparison.
    """
    def __init__(self,
        params,
        nite = 10,
        maxlsiter = 6,
        gradToll = 1e-8,
        alpha = 0.01,
        beta = 0.6,
        t0 = 1,
        maxinitiallsiter=30):
        # Note that I've added an initial maxinitiallsiter > maxlsiter in case the initial step size is much too large.
        defaults = dict(nite=nite,
                        maxlsiter=maxlsiter,
                        gradToll=gradToll,
                        alpha=alpha,
                        beta=beta,
                        t0=t0,
                        maxinitiallsiter=maxinitiallsiter)
        super(CG, self).__init__(params, defaults)

        self._params = self.param_groups[0]['params']

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _line_step(self, p0, step_size, update):
        offset = 0
        for p, p_ref in zip(self._params, p0):
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data = p_ref.data + step_size * update[offset:offset + numel].view_as(p.data)
            offset += numel


    def step(self, closure):
        """
        Run one set of iterations of conjugate gradient descent.

        One call to step runs the algorithm for many internal steps.
        This is similar to the L-BFGS optimizer implemented here:
        https://pytorch.org/docs/stable/_modules/torch/optim/lbfgs.html#LBFGS
        """
        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        eps = 1E-16

        max_iter = group['nite']
        maxlsiter = group['maxlsiter']
        maxinitiallsiter = group['maxinitiallsiter']
        alpha = group['alpha']
        beta = group['beta']
        t0 = group['t0']

        # Right now, we don't track any state from step to step.
        state = self.state[self._params[0]]

        # evaluate initial f(x) and df/dx
        f0 = float(closure(requires_grad=True))
        flat_grad = self._gather_flat_grad()
        g0 = flat_grad.clone()
        dx = -flat_grad

        n_iter = 0

        # Conjugate gradient step, if appropriate.
        while n_iter <= max_iter:
            gd_norm = dx.dot(g0).sum()

            t = t0
            params0 = [x.clone() for x in self._params]
            self._line_step(params0, t, dx)
            loss = closure(requires_grad=False)
            f1 = float(loss)

            # Armijo conditions.
            lsiter = 0
            while (f1 > f0 - alpha * t * abs(gd_norm) and 
                ((n_iter == 0 and lsiter < maxinitiallsiter) or lsiter < maxlsiter)):
                # Backtracking line search.
                lsiter += 1
                t *= beta
                self._line_step(params0, t, dx)
                loss = closure(requires_grad=False)
                f1 = float(loss)

            print("-> %.7f, %.7f, L-S %d" % (f0, f1, lsiter))

            # Heuristics to adapt line search size.
            if n_iter == 0:
                t0 = t
            else:
                if lsiter > 2:
                    t0 *= beta
                elif lsiter < 1:
                    t0 /= beta

            n_iter += 1
            # Call closure again to obtain the gradient.
            loss = closure(requires_grad=True)
            g1 = self._gather_flat_grad()

            bk = (g1.dot(g1)) / (g0.dot(g0) + eps)
            dx = -g1 + bk * dx

            g0 = g1
            f0 = f1

        return loss
