import torch, math

try:
    from apex.optimizers import FusedAdam, FusedLAMB
    from apex.multi_tensor_apply import multi_tensor_applier
    import apex_C, amp_C
except:
    FusedAdam = None
    FusedLAMB = None
    print("To use FusedLAMB or FusedAdam, please install apex.")


class Slamb_V2(torch.optim.Optimizer):
    r"""Implements Lamb algorithm.
    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
        clamp_value: clamp weight_norm in (0,clamp_value) (default: 10)
            set to a high value to avoid it (e.g 10e3)
        adam: always use trust ratio = 1, which turns this
            into Adam. Useful for comparison purposes. (default: False)
        debias: debias adam by (1 - beta**step) (default: False)
    Example:
        >>> optimizer = Lamb(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=0.0,
            clamp_value=10,
            adam=False,
            debias=True,
            max_grad_norm=1.0,
            grad_pre_normalization=False,
            compress_ratio=0.1,
            beta3=0.99,
            local_steps=100,
            c_max=1000,
            c_min=0.01,
            grad_size_thr=9000,

    ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        if clamp_value < 0.0:
            raise ValueError('Invalid clamp value: {}'.format(clamp_value))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, max_grad_norm=max_grad_norm)
        super(Slamb_V2, self).__init__(params, defaults)

        self.clamp_value = clamp_value
        self.adam = adam
        self.debias = debias
        self.grad_pre_normalization = grad_pre_normalization
        self.max_grad_norm = max_grad_norm
        self.global_grad_norm = None

        self.flat_mask = None
        self.mask_counter = {}
        self.mapping = None
        self.flat_grad_size = None
        self.c_max = c_max
        self.c_min = c_min
        self.beta3 = beta3
        self.local_steps = local_steps
        self.compression = 'randomk'
        self.compress_ratio = compress_ratio
        self.global_step = None
        self.world_size = 1
        self.grad_size_thr = grad_size_thr
        self.mask_filter = None
        self.flat_mask_counter = None
        self.flat_m = None
        self.flat_v = None
        self.freshness = None
        self.size_per_param = []

        if multi_tensor_applier.available:
            import amp_C
            self._dummy_overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_l2norm = amp_C.multi_tensor_l2norm
        else:
            raise RuntimeError('apex.contrib.optimizers.FusedLAMB requires cuda extensions')
        print("==INFO== SLAMB optimizer is initialized! Compression ratio: ", self.compress_ratio)

    def sync_moment(self):
        if torch.distributed.get_world_size() > 1:
            self.world_size = torch.distributed.get_world_size()

        flat_sub = torch.masked_select(self.flat_m, self.freshness == 1)
        torch.distributed.all_reduce(flat_sub)
        flat_sub.data.div_(self.world_size)
        self.flat_m[self.freshness == 1] = flat_sub

    def sync_params(self, p_all):
        if torch.distributed.get_world_size() > 1:
            self.world_size = torch.distributed.get_world_size()

        flat_param = torch.cat([p.flatten() for p in p_all])
        torch.distributed.all_reduce(flat_param)
        flat_param.data.div_(self.world_size)

        new_p_all = torch.split(flat_param, self.size_per_param)
        for p, new_p in zip(p_all, new_p_all):
            p.data = new_p.data.view(p.shape)

    def generate_mask(self, p_all):
        if self.freshness is None:
            self.freshness = torch.ones(self.flat_grad_size, dtype=self.grad_dtype, device='cuda')

        if self.compression == 'randomk':
            torch.manual_seed(self.global_step)
            flat_mask = torch.cuda.FloatTensor(self.flat_grad_size).uniform_() < self.compress_ratio

            # always sync low-dim grads
            i = 0
            for p in p_all:
                if p.dim() == 1 or p.numel() < self.grad_size_thr:
                    flat_mask[i:i + p.numel()] = True
                i += p.numel()

            self.freshness.mul_(self.beta3)
            self.freshness[~flat_mask] = 1.0

            del flat_mask

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                p.grad = None

    def step(self, closure=None):
        r"""Performs a single optimization step.
        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        # create separate grad lists for fp32 and fp16 params
        g_all_32, g_all_16 = [], []
        p_all, g_all = [], []

        self.size_per_param = []
        self.grad_dtype = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p_all.append(p)
                g_all.append(p.grad.data)
                self.size_per_param.append(p.numel())
                self.grad_dtype = p.grad.dtype

                if p.dtype == torch.float32:
                    g_all_32.append(p.grad.data)
                elif p.dtype == torch.float16:
                    g_all_16.append(p.grad.data)
                else:
                    raise RuntimeError('FusedLAMB only support fp16 and fp32.')
        self.flat_grad_size = sum([p.numel() for p in p_all])

        if self.grad_pre_normalization:
            g_norm_32, g_norm_16 = 0.0, 0.0
            # compute grad norm for two lists
            if len(g_all_32) > 0:
                g_norm_32 = multi_tensor_applier(self.multi_tensor_l2norm,
                                                 self._dummy_overflow_buf,
                                                 [g_all_32], False)[0].item()
            if len(g_all_16) > 0:
                g_norm_16 = multi_tensor_applier(self.multi_tensor_l2norm,
                                                 self._dummy_overflow_buf,
                                                 [g_all_16], False)[0].item()

            # blend two grad norms to get global grad norm
            global_grad_norm = math.sqrt(g_norm_32 * g_norm_32 + g_norm_16 * g_norm_16)
            self.global_grad_norm = max(global_grad_norm, self.max_grad_norm)

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
            self.global_step = group['step']

            for p in group['params']:
                if p.grad is None:
                    continue

                if p.grad.data.is_sparse:
                    msg = (
                        'Lamb does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                    raise RuntimeError(msg)

                if self.grad_pre_normalization:
                    p.grad.data.div_(self.global_grad_norm)
                state = self.state[p]
                state['weight_decay'] = group['weight_decay']
                # Perform decoupled weight decay
                # if group['weight_decay'] != 0:
                #     p.data.mul_(1 - group['lr'] * group['weight_decay'])

        if self.flat_m is None:
            self.flat_m = torch.zeros(self.flat_grad_size, dtype=torch.float32, device='cuda')
            self.flat_v = torch.zeros(self.flat_grad_size, dtype=torch.float32, device='cuda')

        beta1, beta2 = self.param_groups[0]['betas']
        eps = self.param_groups[0]['eps']
        lr = self.param_groups[0]['lr']

        flat_grad = torch.cat([g.flatten() for g in g_all])
        self.flat_m.mul_(beta1).add_(flat_grad, alpha=1 - beta1)
        self.flat_v.mul_(beta2).addcmul_(flat_grad, flat_grad, value=1 - beta2)
        del flat_grad

        self.generate_mask(p_all)
        self.sync_moment()

        bias_correction1 = 1 - beta1 ** self.global_step
        bias_correction2 = 1 - beta2 ** self.global_step

        flat_adam_step = (self.flat_m / self.flat_v.sqrt().add(eps))
        flat_adam_step.mul_(bias_correction2 ** 0.5 / bias_correction1)
        flat_adam_step *= self.freshness == 1
        tensor_inlist = torch.split(flat_adam_step, self.size_per_param)
        _, max_u = multi_tensor_applier(self.multi_tensor_l2norm, self._dummy_overflow_buf, [tensor_inlist], True)
        del flat_adam_step, tensor_inlist

        flat_adam_step = (self.flat_m / self.flat_v.sqrt().add(eps))
        flat_adam_step.mul_(bias_correction2 ** 0.5 / bias_correction1)
        flat_adam_step *= self.freshness < 1
        tensor_inlist = torch.split(flat_adam_step, self.size_per_param)
        _, min_u = multi_tensor_applier(self.multi_tensor_l2norm, self._dummy_overflow_buf, [tensor_inlist], True)
        del flat_adam_step, tensor_inlist

        flat_param = torch.cat([p.flatten() for p in p_all])
        flat_param *= self.freshness == 1
        tensor_inlist = torch.split(flat_param, self.size_per_param)
        _, max_p = multi_tensor_applier(self.multi_tensor_l2norm, self._dummy_overflow_buf, [tensor_inlist], True)
        del flat_param, tensor_inlist

        flat_param = torch.cat([p.flatten() for p in p_all])
        flat_param *= self.freshness < 1
        tensor_inlist = torch.split(flat_param, self.size_per_param)
        _, min_p = multi_tensor_applier(self.multi_tensor_l2norm, self._dummy_overflow_buf, [tensor_inlist], True)
        del flat_param, tensor_inlist

        max_phi_all = torch.where(
                        (max_p > 0) * (max_u > 0),
                        max_p / max_u,
                        torch.ones_like(max_p)
                    )
        min_phi_all = torch.where(
                        (min_p > 0) * (min_u > 0),
                        min_p / min_u,
                        torch.zeros_like(min_p)
                    )

        max_phi_all = torch.clamp(max_phi_all, self.c_min, self.c_max)
        min_phi_all = torch.clamp(min_phi_all, 0, self.c_max)

        flat_adam_step = (self.flat_m / self.flat_v.sqrt().add(eps))
        flat_adam_step.mul_(bias_correction2 ** 0.5 / bias_correction1)
        u_all = torch.split(flat_adam_step, self.size_per_param)
        r_all = torch.split(self.freshness, self.size_per_param)

        for p, u, max_phi, min_phi, r in zip(p_all, u_all, list(max_phi_all), list(min_phi_all), r_all):
            u = u.view(p.shape)
            r = r.view(p.shape)
            lr_min = lr / math.sqrt(self.world_size)
            state = self.state[p]
            if state['weight_decay'] != 0:
                u.data.add_(p.data, alpha=state['weight_decay'])
            p.data.add_(- lr * max_phi * u * r - lr_min * min_phi * u * (1 - r))

            # new_lr = r * lr + (1 - r) * lr_min
            # new_phi = r * max_phi + (1 - r) * min_phi
            # p.data.add_(- new_lr * new_phi * u)

        if self.global_step % self.local_steps == 1:
            self.sync_params(p_all)
        return loss