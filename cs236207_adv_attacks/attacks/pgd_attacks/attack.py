import torch
from torch.nn import functional as F


class Attack:
    def __init__(self, model, criterion, misc_args, pgd_args, name="PGD"):
        self.model = model
        self.criterion = criterion

        self.name = name
        self.device = misc_args['device']
        self.dtype = misc_args['dtype']
        self.batch_size = misc_args['batch_size']
        self.data_shape = [self.batch_size] + misc_args['data_shape']
        self.data_channels = self.data_shape[1]
        self.data_w = self.data_shape[2]
        self.data_h = self.data_shape[3]
        self.n_data_pixels = self.data_w * self.data_h
        self.data_RGB_start = (torch.tensor(misc_args['data_RGB_start'], device=self.device).
                                 unsqueeze(0).unsqueeze(2).unsqueeze(3))
        self.data_RGB_end = (torch.tensor(misc_args['data_RGB_end'], device=self.device).
                               unsqueeze(0).unsqueeze(2).unsqueeze(3))
        self.data_RGB_size = (torch.tensor(misc_args['data_RGB_size'], device=self.device).
                               unsqueeze(0).unsqueeze(2).unsqueeze(3))
        self.verbose = misc_args['verbose']
        self.report_info = misc_args['report_info']

        self.norm = pgd_args['norm']
        self.p = float(self.norm[1:])
        self.pert_lb = None
        self.pert_ub = None
        self.eps_ratio = pgd_args['eps']
        self.eps = self.eps_ratio * self.data_RGB_size
        self.n_restarts = pgd_args['n_restarts']
        self.n_iter = pgd_args['n_iter']
        self.alpha = pgd_args['alpha'] * self.data_RGB_size
        if self.alpha is None:
            self.alpha = self.eps / self.n_iter
        self.a_abs = self.alpha.abs()
        self.rand_init = pgd_args['rand_init']
        self.targeted_mul = None
        self.multiplier = None
        self.eval_pert = None
        self.clean_loss = None
        self.clean_succ = None

    def set_params(self, x, targeted):
        self.batch_size = x.shape[0]
        self.data_shape[0] = x.shape[0]
        self.set_multiplier(targeted)
        if self.name == 'PGD':
            self.pert_lb = self.data_RGB_start - x
            self.pert_ub = self.data_RGB_end - x
        elif self.name == 'UPGD':
        # use the minimum and maximum over the batch
            # self.pert_lb = (self.data_RGB_start - x).min(dim=0)[0].unsqueeze(0)
            # self.pert_ub = (self.data_RGB_end - x).max(dim=0)[0].unsqueeze(0)
            self.pert_lb = self.data_RGB_start - x
            self.pert_ub = self.data_RGB_end - x

    def set_multiplier(self, targeted):
        if targeted:
            self.targeted_mul = -1
            self.eval_pert = self.eval_pert_targeted
        else:
            self.targeted_mul = 1
            self.eval_pert = self.eval_pert_untargeted
        self.multiplier = (self.targeted_mul * self.a_abs).to(self.device)

    def random_initialization(self):
        if self.norm == 'Linf':
            return torch.empty(self.data_shape, dtype=self.dtype, device=self.device).uniform_(-1, 1) * self.eps
        else:
            return torch.empty(self.data_shape, dtype=self.dtype, device=self.device).normal_(0, self.eps * self.eps)

    def project(self, perturbation):
        if self.name == 'PGD':
            if self.norm == 'Linf':
                pert = torch.clamp(perturbation, -self.eps, self.eps)
            else:
                pert = F.normalize(perturbation.view(perturbation.shape[0], -1),
                                p=self.p, dim=-1).view(perturbation.shape) * self.eps
            pert.clamp_(self.pert_lb, self.pert_ub)

        elif self.name == 'UPGD':
            if self.norm == 'Linf':
                pert = torch.clamp(perturbation, -self.eps, self.eps)
            else:
                pert = F.normalize(perturbation.view(perturbation.shape[0], -1),
                                p=self.p, dim=-1).view(perturbation.shape) * self.eps
            # pert.clamp_(self.pert_lb, self.pert_ub)

        return pert

    def normalize_grad(self, grad):
        if self.norm == 'Linf':
            return grad.sign()
        else:
            return F.normalize(grad.view(grad.shape[0], -1), p=self.p, dim=-1).view(grad.shape)

    def step(self, pert, grad):
        grad = self.normalize_grad(grad)
        pert += self.multiplier * grad
        return self.project(pert)

    def test_pert(self, x, y, pert):
        with torch.no_grad():
            output = self.model.forward(x + pert)
            loss = self.targeted_mul * self.criterion(output, y)
            return output, loss

    def eval_pert_untargeted(self, x, y, pert):
        with torch.no_grad():
            output, loss = self.test_pert(x, y, pert)
            succ = torch.argmax(output, dim=1) != y
            return loss, succ

    def eval_pert_targeted(self, x, y, pert):
        with torch.no_grad():
            output, loss = self.test_pert(x, y, pert)
            succ = torch.argmax(output, dim=1) == y
            return loss, succ

    def update_best(self, best_crit, new_crit, best_ls, new_ls):
        improve = new_crit > best_crit
        best_crit[improve] = new_crit[improve]
        for idx, best in enumerate(best_ls):
            new = new_ls[idx]
            best[improve] = new[improve]

    def perturb(self, x, y, targeted=False):
        raise NotImplementedError('You need to define a perturb method!')
