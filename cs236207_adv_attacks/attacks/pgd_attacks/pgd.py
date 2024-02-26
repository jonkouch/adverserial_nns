import torch
from attacks.pgd_attacks.attack import Attack


class PGD(Attack):
    def __init__(
            self,
            model,
            criterion,
            misc_args=None,
            pgd_args=None):
        super(PGD, self).__init__(model, criterion, misc_args, pgd_args)

    def report_schematics(self):

        print("Attack L_inf norm limitation:")
        print(self.eps_ratio)
        print("Number of iterations for perturbation optimization:")
        print(self.n_iter)
        print("Number of restarts for perturbation optimization:")
        print(self.n_restarts)

    def perturb(self, x, y, targeted=False):
        with torch.no_grad():
            self.set_params(x, targeted)
            self.clean_loss, self.clean_succ = self.eval_pert(x, y, pert=torch.zeros_like(x))
            best_pert = torch.zeros_like(x)
            best_loss = self.clean_loss.clone().detach()
            best_succ = self.clean_succ.clone().detach()

            if self.report_info:
                all_best_succ = torch.zeros(self.n_restarts,  self.n_iter + 1, self.batch_size, dtype=torch.bool, device=self.device)
                all_best_loss = torch.zeros(self.n_restarts,  self.n_iter + 1, self.batch_size, dtype=self.dtype, device=self.device)
            else:
                all_best_succ = None
                all_best_loss = None

        self.model.eval()
        for rest in range(self.n_restarts):
            if self.rand_init:
                pert_init = self.random_initialization()
                pert_init = self.project(pert_init)
            else:
                pert_init = torch.zeros_like(x,)

            with torch.no_grad():
                loss, succ = self.eval_pert(x, y, pert_init)
                self.update_best(best_loss, loss,
                                 [best_pert, best_succ],
                                 [pert_init, succ])
                if self.report_info:
                    all_best_succ[rest, 0] = best_succ
                    all_best_loss[rest, 0] = best_loss

            pert = pert_init.clone().detach()
            for k in range(1, self.n_iter + 1):
                pert.requires_grad_()
                train_loss = self.criterion(self.model.forward(x + pert), y)
                grad = torch.autograd.grad(train_loss.mean(), [pert])[0].detach()

                with torch.no_grad():
                    pert = self.step(pert, grad)
                    eval_loss, succ = self.eval_pert(x, y, pert)
                    self.update_best(best_loss, eval_loss,
                                     [best_pert, best_succ],
                                     [pert, succ])

                if self.report_info:
                    all_best_succ[rest, k] = best_succ
                    all_best_loss[rest, k] = best_loss

        adv_pert = best_pert.clone().detach()
        adv_pert_loss = best_loss.clone().detach()
        return adv_pert, adv_pert_loss, all_best_succ, all_best_loss
