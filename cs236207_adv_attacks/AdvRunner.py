import numpy as np
import torch
from tqdm import trange


class AdvRunner:
    def __init__(self, model, attack, data_RGB_size, device, dtype, verbose=False):
        self.attack = attack
        self.model = model
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.data_RGB_size = torch.tensor(data_RGB_size).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.attack_restarts = self.attack.n_restarts
        self.attack_iter = self.attack.n_iter
        self.attack_report_info = self.attack.report_info
        self.attack_name = self.attack.name

    def run_clean_evaluation(self, x_orig, y_orig, n_examples, bs, n_batches, orig_device):
        robust_flags = torch.zeros(n_examples, dtype=torch.bool, device=orig_device)
        for batch_idx in range(n_batches):
            start_idx = batch_idx * bs
            end_idx = min((batch_idx + 1) * bs, n_examples)

            x = x_orig[start_idx:end_idx, :].clone().detach().to(self.device)
            y = y_orig[start_idx:end_idx].clone().detach().to(self.device)
            output = self.model.forward(x)
            correct_batch = y.eq(output.max(dim=1)[1]).detach().to(orig_device)
            robust_flags[start_idx:end_idx] = correct_batch

        n_robust_examples = torch.sum(robust_flags).item()
        init_accuracy = n_robust_examples / n_examples
        if self.verbose:
            print('initial accuracy: {:.2%}'.format(init_accuracy))
        return robust_flags, n_robust_examples, init_accuracy

    def process_results(self, n_examples, robust_flags, adv_perts, adv_perts_loss):

        robust_accuracy = (robust_flags.sum(dim=0) / n_examples).item()
        if self.attack_name == 'PGD':
            adv_loss = adv_perts_loss.mean(dim=0).item()
        else:
            adv_loss = adv_perts_loss
        adv_perts = adv_perts.to('cpu')
        perts_max_l_inf = (adv_perts.abs() / self.data_RGB_size).view(-1).max(dim=0)[0].item()
        return robust_accuracy, adv_loss, perts_max_l_inf

    def run_standard_evaluation(self, x_orig, y_orig, n_examples, bs=250):
        with torch.no_grad():
            orig_device = x_orig.device
            # calculate accuracy
            n_batches = int(np.ceil(n_examples / bs))
            robust_flags, n_robust_examples, init_accuracy = self.run_clean_evaluation(x_orig, y_orig, n_examples, bs, n_batches, orig_device)

        x_adv = x_orig.clone().detach()
        y_adv = y_orig.clone().detach()
        adv_perts = torch.zeros_like(x_orig).detach()
        adv_perts_loss = torch.zeros(n_examples, dtype=self.dtype, device=orig_device)
        if self.attack_report_info:
            info_shape = [self.attack_restarts, self.attack_iter + 1, n_examples]
            all_succ = torch.zeros(info_shape, dtype=self.dtype, device=orig_device)
            all_loss = torch.zeros(info_shape, dtype=self.dtype, device=orig_device)
        else:
            all_succ = None
            all_loss = None
        with torch.cuda.device(self.device):
            start_events = [torch.cuda.Event(enable_timing=True) for batch_idx in range(n_batches)]
            end_events = [torch.cuda.Event(enable_timing=True) for batch_idx in range(n_batches)]

            if self.attack_name == 'PGD':
                for batch_idx in trange(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, n_examples)
                    batch_indices = torch.arange(start_idx, end_idx, device=orig_device)
                    x = x_orig[start_idx:end_idx, :].clone().detach().to(self.device)
                    y = y_orig[start_idx:end_idx].clone().detach().to(self.device)

                    # make sure that x is a 4d tensor even if there is only a single datapoint left
                    if len(x.shape) == 3:
                        x.unsqueeze_(dim=0)
                    start_events[batch_idx].record()
                    batch_adv_perts, batch_adv_perts_loss, all_batch_succ, all_batch_loss = self.attack.perturb(x, y)
                    end_events[batch_idx].record()
                    torch.cuda.empty_cache()
                    with torch.no_grad():

                        batch_x_adv = x + batch_adv_perts
                        x_adv[start_idx:end_idx] = batch_x_adv.detach().to(orig_device)
                        output = self.model.forward(batch_x_adv)
                        batch_y_adv = output.max(dim=1)[1]
                        y_adv[start_idx:end_idx] = batch_y_adv.to(orig_device)
                        adv_perts[start_idx:end_idx] = batch_adv_perts.detach().to(orig_device)
                        adv_perts_loss[start_idx:end_idx] = batch_adv_perts_loss.detach().to(orig_device)
                        false_batch = ~y.eq(batch_y_adv).detach().to(orig_device)
                        non_robust_indices = batch_indices[false_batch]
                        robust_flags[non_robust_indices] = False
                        if self.attack_report_info:
                            all_succ[:, :, start_idx:end_idx] = all_batch_succ.to(orig_device)
                            all_loss[:, :, start_idx:end_idx] = all_batch_loss.to(orig_device)

            else: 
                #read this
                pert = self.attack.perturb()
                # set all entries of adv_perts as pert
                adv_perts = pert.repeat(n_examples, 1, 1, 1)
                adv_perts_loss = self.attack.loss

                # change the robust flags where the attack was successful
                adv_perts = adv_perts.to('cpu')
                x_adv = x_orig + adv_perts
                # change the robust flags where the attack was successful
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, n_examples)
                    batch_indices = torch.arange(start_idx, end_idx, device=orig_device)
                    x = x_adv[start_idx:end_idx, :].clone().detach().to(self.device)
                    y = y_orig[start_idx:end_idx].clone().detach().to(self.device)
                    output = self.model.forward(x)
                    batch_y_adv = output.max(dim=1)[1]
                    y_adv[start_idx:end_idx] = batch_y_adv.to(orig_device)
                    false_batch = ~y.eq(batch_y_adv).detach().to(orig_device)
                    non_robust_indices = batch_indices[false_batch]
                    robust_flags[non_robust_indices] = False
                
            if self.attack_name == 'PGD':
                torch.cuda.synchronize()
                adv_batch_compute_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
                adv_batch_compute_time_mean = np.mean(adv_batch_compute_times)
                adv_batch_compute_time_std = np.std(adv_batch_compute_times)
                tot_adv_compute_time = np.sum(adv_batch_compute_times)
                tot_adv_compute_time_std = np.std([time * n_batches for time in adv_batch_compute_times])
            else:
                adv_batch_compute_time_mean, adv_batch_compute_time_std, tot_adv_compute_time, tot_adv_compute_time_std = None, None, None, None

        with torch.no_grad():
            robust_accuracy, adv_loss, perts_max_l_inf = \
                self.process_results(n_examples, robust_flags, adv_perts, adv_perts_loss)

            if self.verbose:
                print("reporting results for adversarial attack: " + self.attack_name)
                if self.attack_name == 'PGD':
                    print("Attack batches runtime mean: " + str(adv_batch_compute_time_mean) + " s")
                    print("Attack batches runtime std: " + str(tot_adv_compute_time_std) + " s")
                    print("Attack total runtime: " + str(tot_adv_compute_time) + " s")
                    print("Attack total runtime std over batches: " + str(tot_adv_compute_time_std) + " s")
                print("clean accuracy:")
                print(init_accuracy)
                print("robust accuracy:")
                print(robust_accuracy)
                print("perturbations max L_inf:")
                print(perts_max_l_inf)
                print('nan in tensors: {}, max: {:.5f}, min: {:.5f}'.format(
                    (adv_perts != adv_perts).sum(), adv_perts.max(),
                    adv_perts.min()))
            if self.attack_report_info:
                acc_steps = 1 - (all_succ.sum(dim=2) / n_examples)
                avg_loss_steps = all_loss.sum(dim=2) / n_examples
            else:
                acc_steps = None
                avg_loss_steps = None
            del all_succ
            del all_loss
            del adv_perts
            torch.cuda.empty_cache()



            return init_accuracy, x_adv, y_adv, robust_accuracy, adv_loss, \
                acc_steps, avg_loss_steps, perts_max_l_inf, \
                adv_batch_compute_time_mean, adv_batch_compute_time_std, tot_adv_compute_time, tot_adv_compute_time_std

