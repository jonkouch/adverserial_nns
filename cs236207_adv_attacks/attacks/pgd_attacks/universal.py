import torch
from attacks.pgd_attacks.attack import Attack
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time

class UPGD(Attack):
    def __init__(
            self,
            model,
            criterion,
            misc_args=None,
            pgd_args=None,
            x_orig=None,
            y_orig=None,
            model_name=""
            ):
        super(UPGD, self).__init__(model, criterion, misc_args, pgd_args, name="UPGD")

        self.model_name = model_name
        self.pert = self.train_attack1(x_orig, y_orig)
        # self.pert = self.train_targeted_attack(x_orig, y_orig, 9)
        self.loss = self.compute_loss(x_orig, y_orig, self.pert)
        self.adv_accuracy = self.compute_accuracy(x_orig, y_orig, self.pert)


    def train_attack1(self, x_orig, y_orig):
        if not isinstance(x_orig, torch.Tensor):
            x_orig = torch.tensor(x_orig, dtype=torch.float32)
        if not isinstance(y_orig, torch.Tensor):
            y_orig = torch.tensor(y_orig, dtype=torch.long)

        dataset = TensorDataset(x_orig, y_orig)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        curr_pert = torch.zeros_like(x_orig[0]).to(self.device)
        curr_pert.requires_grad = True

        optimizer = torch.optim.AdamW([curr_pert], lr=0.1)
        # Using CosineAnnealingLR for learning rate adjustment
        scheduler = CosineAnnealingLR(optimizer, T_max=self.n_iter, eta_min=0.001)  # Adjust T_max and eta_min as needed

        batch_count = 0
        update_freq = 5

        accuracies = []
        start_time = time.time()
        pbar = tqdm(total=self.n_iter)
        best_state = None
        best_acc = 1
        best_epoch = 0
        for epoch in range(self.n_iter):
            correct = 0
            total = 0
            epoch_loss = 0

            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                self.model.zero_grad()  # Zero gradients for model

                with torch.no_grad():
                    self.set_params(x_batch, False)

                x_perturbed = x_batch + curr_pert
                x_perturbed = torch.clamp(x_perturbed, 0, 1)

                output = self.model(x_perturbed)

                # Mask to select correctly classified examples
                correct_preds = output.argmax(1) == y_batch
                if correct_preds.any():
                    loss = -self.criterion(output[correct_preds], y_batch[correct_preds]).mean()
                    epoch_loss += loss.item()
                    loss.backward()

                    if (batch_count + 1) % (1+epoch//update_freq) == 0:
                        optimizer.step()  # Update the perturbation
                        optimizer.zero_grad()  # Zero gradients for optimizer

                        with torch.no_grad():
                            # Optionally project the perturbation to ensure it stays within desired bounds after the update
                            curr_pert[:] = self.project(curr_pert)
                            curr_pert.requires_grad = True  # Re-enable gradient tracking

                batch_count += 1
                correct += correct_preds.sum().item()
                total += y_batch.size(0)

            accuracies.append(correct / total)
            epoch_acc = correct / total

            scheduler.step()  # Update learning rate

            if best_state is None or epoch_acc < best_acc:
                best_state = curr_pert.clone().detach()
                best_acc = epoch_acc
                best_epoch = epoch

            pbar.set_description(f"Epoch {epoch + 1}/{self.n_iter}, Loss: {epoch_loss / total:.4f}, Acc: {accuracies[-1] * 100:.2f}%, LR: {scheduler.get_last_lr()[0]:.5f}")
            pbar.update()
        pbar.close()
        end_time = time.time()
        print(f'Elapsed time for training the attack: {end_time - start_time}')
        print(f'Average time per epoch: {(end_time - start_time) / self.n_iter}')

        if self.model_name == 'ResNet18':
            title = 'standard'
        else:
            title = 'robust'


        plt.plot(accuracies)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'UPGD Accuracy over epochs - {title} model')
        # save the plot
        path = f'results/plots/upgd_accuracy_{title}.png'
        plt.savefig(path)
        
        path = f'results/perts/pert_upgd_{title}.pt'
        torch.save(curr_pert, path)
        torch.save(best_state, f'results/perts/best_pert_upgd_{title}_{best_epoch}.pt')

        return curr_pert

    
    def train_targeted_attack(self, x_orig, y_orig, target):
        """
        Train a targeted attack
        :param x_orig: input batch
        :param y_orig: target batch
        :param target: target label
        """
        if not isinstance(x_orig, torch.Tensor):
            x_orig = torch.tensor(x_orig, dtype=torch.float32)
        if not isinstance(y_orig, torch.Tensor):
            y_orig = torch.tensor(y_orig, dtype=torch.long)

        # Create a dataset and loader for the original data
        dataset = TensorDataset(x_orig, y_orig)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize perturbation
        curr_pert = torch.zeros_like(x_orig[0]).to(self.device)
        curr_pert.requires_grad = True

        # Optimizer for the perturbation
        optimizer = torch.optim.AdamW([curr_pert], lr=0.01)

        accuracies = []
        start_time = time.time()
        batch_count = 0
        for epoch in tqdm(range(self.n_iter)):
            correct = 0
            total = 0
            for x_batch, y_batch in loader:  # Ignore y_orig here as we're targeting a specific class
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.model.zero_grad()  # Zero gradients for model

                x_perturbed = x_batch + curr_pert
                x_perturbed = torch.clamp(x_perturbed, 0, 1)  # Ensure perturbed image is valid

                output = self.model(x_perturbed)
                # Create a tensor of the target labels, anc change the target label to any other label
                target_labels = torch.full_like(y_batch, target).to(self.device)
                change_indices = torch.where(target_labels == y_batch)
                target_labels[change_indices] = (target_labels[change_indices] + 1) % 10

                total += y_batch.size(0)
                correct += (output.argmax(1) == y_batch).sum().item()
                
                loss = -self.criterion(output, target_labels).mean()
                loss.backward()

                if (batch_count + 1) % (1+epoch//10) == 0:
                    optimizer.step()  # Update the perturbation
                    optimizer.zero_grad()  # Zero gradients for optimizer

                    with torch.no_grad():
                        # Optionally project the perturbation to ensure it stays within desired bounds after the update
                        curr_pert[:] = self.project(curr_pert)
                        curr_pert.requires_grad = True  # Re-enable gradient tracking
                
                batch_count += 1

            
            accuracies.append(correct / total)

        # Optionally, log or save results as done in train_attack1
        path = f'results/pert_targeted_upgd_{self.model_name}_target{target}.pt'
        torch.save(curr_pert, path)

        end_time = time.time()
        print(f'Elapsed time for training the attack: {end_time - start_time}')
        print(f'Average time per epoch: {(end_time - start_time) / self.n_iter}')

        if self.model_name == 'ResNet18':
            title = 'standard'
        else:
            title = 'robust'


        plt.plot(accuracies)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'UPGD Accuracy over epochs - {title} model')
        # save the plot
        path = f'results/plots/upgd_accuracy_{title} - target{target}.png'
        plt.savefig(path)
        
        path = f'results/perts/pert_targeted_upgd_{title}_target{target}.pt'
        torch.save(curr_pert, path)
        return curr_pert

    def train_attack_epgd(self, x_orig, y_orig, num_samples=5):
        if not isinstance(x_orig, torch.Tensor):
            x_orig = torch.tensor(x_orig, dtype=torch.float32)
        if not isinstance(y_orig, torch.Tensor):
            y_orig = torch.tensor(y_orig, dtype=torch.long)

        dataset = TensorDataset(x_orig, y_orig)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        curr_pert = torch.zeros_like(x_orig[0]).to(self.device)
        curr_pert.requires_grad = True

        optimizer = torch.optim.Adam([curr_pert], lr=0.01)

        update_freq = 5
        batch_count = 0
        epsilon = self.eps/50

        accuracies = []
        for epoch in tqdm(range(self.n_iter)):
            correct = 0
            total = 0
            grad_sum = 0
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                self.model.zero_grad()

                with torch.no_grad():
                    self.set_params(x_batch, False)

                avg_grad = None
                for _ in range(num_samples):
                    # Sample within epsilon ball around the original image
                    random_pert = (2 * epsilon * torch.rand_like(x_batch) - epsilon).to(self.device)
                    x_perturbed = x_batch + random_pert
                    x_perturbed = torch.clamp(x_perturbed + curr_pert, 0, 1)

                    output = self.model(x_perturbed)
                    loss = -self.criterion(output, y_batch).mean()
                    loss.backward()

                    if avg_grad is None:
                        avg_grad = curr_pert.grad.clone()
                    else:
                        avg_grad += curr_pert.grad

                    self.model.zero_grad()
                    curr_pert.grad.zero_()

                avg_grad /= num_samples
                grad_sum += avg_grad


                if (batch_count + 1) % update_freq == 0:
                    curr_pert.grad = grad_sum / update_freq
                    grad_sum = 0
                    optimizer.step()
                    optimizer.zero_grad()

                    with torch.no_grad():
                        curr_pert[:] = self.project(curr_pert)
                        curr_pert.requires_grad = True
                    
                batch_count += 1
                correct += (output.argmax(1) == y_batch).sum().item()
                total += y_batch.size(0)

            accuracies.append(correct / total)

        if self.model_name == '':
            title = 'standard'
        else:
            title = 'robust'

        plt.plot(accuracies)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'EPGD Accuracy over epochs - {title} model')
        plt.savefig(f'results/epgd_accuracy_{title}.png')

        path = f'results/pert_epgd_{title}.pt'
        torch.save(curr_pert, path)
        return curr_pert


    def report_schematics(self):

        print("Attack L_inf norm limitation:")
        print(self.eps_ratio)
        print("Number of iterations for perturbation optimization:")
        print(self.n_iter)
        print("Number of restarts for perturbation optimization:")
        print(self.n_restarts)


    def compute_loss(self, x, y, pert):
        with torch.no_grad():
            pert = pert.to(self.device)
            loss = 0 
            dataset = TensorDataset(x, y)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # get the max and min values of the perturbed data
            max_val = -1
            min_val = 1
            flag = False
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                x_perturbed = x_batch + pert
                x_perturbed = torch.clamp(x_perturbed, 0, 1)

                # save an image before and after the perturbation
                img = x_batch[0].cpu().detach().numpy()
                img_pert = x_perturbed[0].cpu().detach().numpy()
                img = img.transpose(1, 2, 0)
                img_pert = img_pert.transpose(1, 2, 0)
                img = Image.fromarray((img * 255).astype(np.uint8))
                img_pert = Image.fromarray((img_pert * 255).astype(np.uint8))
                
                if not flag:
                    if self.model_name == 'ResNet18':
                        title = 'standard'
                    else:
                        title = 'robust'
                    
                    img.save(f'results/clean_img_{title}.png')
                    img_pert.save(f'results/pert_img_{title}.png')
                    flag = True

                output = self.model(x_perturbed)
                loss += self.criterion(output, y_batch).mean().item()
                if max_val < x_perturbed.max():
                    max_val = x_perturbed.max()
                if min_val > x_perturbed.min():
                    min_val = x_perturbed.min()
            loss /= len(loader)

        return loss
    
    def compute_accuracy(self, x, y, pert):
        with torch.no_grad():
            pert = pert.to(self.device)
            correct = 0
            total = 0
            dataset = TensorDataset(x, y)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                x_perturbed = x_batch + pert
                x_perturbed = self.project(x_perturbed)
                output = self.model(x_perturbed)
                _, predicted = torch.max(output, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
            accuracy = correct / total

        return accuracy
    
    def get_non_robust_indices(self, x_adv, y, robust_flags):
        for batch_idx in range(self.n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, self.n_examples)

            x = x_adv[start_idx:end_idx, :].clone().detach().to(self.device)
            y = y[start_idx:end_idx].clone().detach().to(self.device)
            output = self.model.forward(x)
            correct_batch = y.eq(output.max(dim=1)[1]).detach().to(self.device)
            robust_flags[start_idx:end_idx] = correct_batch
        return robust_flags
            

    def perturb(self, targeted=False):
        """
        
        Universal 
        :param x: input batch
        :param y: target batch
        :param targeted: targeted attack flag
        :return: adversarial perturbation
        """

        return self.pert

        