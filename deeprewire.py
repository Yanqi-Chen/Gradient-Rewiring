import torch
from torch.optim.optimizer import Optimizer, required
import math

# NOTE: The parameter T (tempertature) here corresponds to the sqrt(2*T) in paper of DEEP R

class DeepRewiring(Optimizer):
    """
    Based on the implementation of Adam optimizer in PyTorch and offcial repo of DEEP R (https://github.com/guillaumeBellec/deep_rewiring).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, T=0, l1=1e-5, max_s=None, soft=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= max_s <= 1.0:
            raise ValueError("Invalid target sparsity: {}, must be between 0 and 1".format(s))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, T=T, l1=l1, max_s=max_s, soft=soft)
        super(DeepRewiring, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DeepRewiring, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    # Record initial sign
                    state['sign'] = torch.sign(p)

                    if not group['soft']:
                        state['numel'] = p.numel()
                    else:
                        # Hidden parameter theta
                        state['strength'] = torch.abs(p)
                        if group['l1'] > 0 and group['max_s'] < 1:
                            # θ_min = -T(1-p_min)/(α p_min)
                            state['strength_min'] = -(0.5 * (group['T'] ** 2) * group['max_s']) / (group['l1'] * (1 - group['max_s']))
                    
                    state['active'] = (p != 0.0).float()

                sgn = state['sign']

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                if group['soft']: # Soft-Deep R

                    p_hidden = sgn * state['strength']

                    # Gradient term
                    p_hidden.addcdiv_(exp_avg * state['active'], denom, value=-step_size)

                    # L1 penalty
                    p_hidden.addcmul_(sgn, state['active'], value=-group['l1'] * step_size)

                    # Noise term
                    rand_normal = torch.randn_like(p)
                    p_hidden.add_(rand_normal, alpha=group['T'] * math.sqrt(step_size))

                    state['strength'] = p_hidden * sgn
                    if group['l1'] > 0 and group['max_s'] < 1:   
                        state['strength'].clamp_(min=state['strength_min'])

                    # Prune those connection changing their signs
                    p.data = state['strength'].clamp(min=0.0).mul(sgn)

                else: # Deep R

                    # Gradient term
                    p.addcdiv_(exp_avg * state['active'], denom, value=-step_size)

                    # L1 penalty
                    p.addcmul_(sgn, state['active'], value=-group['l1'] * step_size)

                    # Noise term
                    rand_normal = torch.randn_like(p)
                    p.addcmul_(rand_normal, state['active'], value=group['T'] * math.sqrt(step_size))

                    # Prune those connection changed their signs
                    p.mul_(state['active']).mul_(sgn).clamp_(min=0.0).mul_(sgn)

                state['active'] = (p != 0.0).float()

                if not group['soft']: # Deep R
                    dormant = 1 - state['active']
                    curr_s = dormant.sum() / state['numel']

                    # Activate connections when number of dormant is more than expected
                    if curr_s > group['max_s']:
                        rewire_prob = (curr_s - group['max_s']) / curr_s
                        rewire_mask = (dormant * rewire_prob).bernoulli().bool() # The probability of rewiring is dormant * rewire_prob
                        p.data.masked_fill_(rewire_mask, group['eps']) # Hard Rewire

        return loss