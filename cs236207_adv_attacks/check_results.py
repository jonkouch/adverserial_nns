import numpy as np
import torch
from models.cifar10.resnet import ResNet18
from robustbench.data import load_cifar10
from robustbench.utils import load_model


def main():
    eps = 8/255
    pert_path = 'results/pert_upgd_standard.pt'
    pert = torch.load(pert_path)

    # check if the perturbation is within the L_inf bound and normalize if necessary
    if torch.abs(pert).max() > eps:
        print('Perturbation exceeds L_inf bound - normalizing')
        pert = torch.clamp(pert, -eps, eps)

    # make sure pert dimension is (1, 3, 32, 32)
    if len(pert.shape) == 3:
        pert = pert.unsqueeze(0)
    if pert.shape != (1, 3, 32, 32):
        raise ValueError('Perturbation has incorrect dimensions')

    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # load the ResNet18 model
    if 'standard' in pert_path:
        state_path = 'models/cifar10/resnet18.pt'
        model = ResNet18(device)
        model.load_state_dict(torch.load(state_path))
    # load the robust model
    else:
        model = load_model(model_name='Wong2020Fast', dataset='cifar10', threat_model='Linf').to(device)
    
    # load the CIFAR10 dataset
    data_dir = 'data'
    n_examples = 10000
    x_test, y_test = load_cifar10(n_examples=n_examples, data_dir=data_dir)
    bs = 250
    n_batches = int(np.ceil(n_examples / bs))

    # run clean evaluation
    robust_flags = torch.zeros(n_examples, dtype=torch.bool)
    model.eval()
    print('Running clean evaluation:')
    for batch_idx in range(n_batches):
        start_idx = batch_idx * bs
        end_idx = min((batch_idx + 1) * bs, n_examples)

        x = x_test[start_idx:end_idx, :].clone().detach().to(device)
        y = y_test[start_idx:end_idx].clone().detach().to(device)
        output = model.forward(x)
        correct_batch = y.eq(output.max(dim=1)[1]).detach()
        robust_flags[start_idx:end_idx] = correct_batch

    n_robust_examples = torch.sum(robust_flags).item()
    init_accuracy = n_robust_examples / n_examples
    print('initial accuracy: {:.2%}'.format(init_accuracy))

    # run adversarial attack
    print('Running evaluation of adversarial attack:')
    for batch_idx in range(n_batches):
        start_idx = batch_idx * bs
        end_idx = min((batch_idx + 1) * bs, n_examples)
        batch_indices = torch.arange(start_idx, end_idx, device=device)
        x = x_test[start_idx:end_idx, :].clone().detach().to(device)
        y = y_test[start_idx:end_idx].clone().detach().to(device)
        x = x + pert
        x = torch.clamp(x, 0, 1)
        output = model.forward(x).max(dim=1)[1]
        false_batch = ~y.eq(output).detach()
        non_robust_indices = batch_indices[false_batch]
        robust_flags[non_robust_indices] = False

    n_robust_examples = torch.sum(robust_flags).item()
    robust_accuracy = n_robust_examples / n_examples
    adv_succ_ratio = (init_accuracy - robust_accuracy) / init_accuracy
    print('robust accuracy: {:.2%}'.format(robust_accuracy))
    print('adversarial attack success ratio: {:.2%}'.format(adv_succ_ratio))


if __name__ == '__main__':
    main()
    