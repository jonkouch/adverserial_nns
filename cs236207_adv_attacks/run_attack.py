import torch
import gc
from parser_1 import get_args, save_img_tensors
from AdvRunner import AdvRunner


def run_adv_attacks(args):
    print(f'Running evaluation of adversarial attacks:')
    adv_runner = AdvRunner(args.model, args.attack_obj, args.data_RGB_size,
                           device=args.device, dtype=args.dtype, verbose=args.runner_verbose)
    print(f'Dataset: {args.dataset}, Model: {args.model_name},\n'
          f'Attack: {args.attack_name} with L_inf epsilon={args.eps_l_inf},\n'
          f'Attack iterations={args.n_iter} and restarts={args.n_restarts}')
    print("Shape of input samples:")
    print(args.data_shape)
    print("Data RGB range:")
    print(list(zip(args.data_RGB_start, args.data_RGB_end)))

    args.attack_obj.report_schematics()
    att_report_info = args.attack_obj.report_info

    (init_accuracy, x_adv, y_adv, robust_accuracy, adv_loss,
     acc_steps, avg_loss_steps, perts_max_l_inf,
     adv_batch_compute_time_mean, adv_batch_compute_time_std, tot_adv_compute_time, tot_adv_compute_time_std) = \
        adv_runner.run_standard_evaluation(args.x_test, args.y_test, args.n_examples, bs=args.batch_size)

    adv_succ_ratio = (init_accuracy - robust_accuracy) / init_accuracy
    print("reporting results for adversarial attack on Model: " + args.model_name)
    print("attacked model clean accuracy:")
    print(init_accuracy)
    print(f'robust accuracy: {robust_accuracy}')
    print(f'adversarial attack success ratio: {adv_succ_ratio}')
    print(f'adversarial average loss: {adv_loss}')
    print(f'perturbations L_inf norm limitation : {args.eps_l_inf}')
    print(f'max L_inf in perturbations: {perts_max_l_inf}')
    print("attack mean compute time over data batches")
    print(adv_batch_compute_time_mean)
    print("attack std compute time over data batches")
    print(adv_batch_compute_time_std)
    print("attack total compute time")
    print(tot_adv_compute_time)
    print("attack total compute time std over data batches")
    print(tot_adv_compute_time_std)

    if att_report_info:
        adv_succ_ratio_steps = [(init_accuracy - adv_acc) / init_accuracy for adv_acc in acc_steps]
        print("reporting the optimization info of the attack:")
        print(f'Model: {args.model_name}, adversarial robust accuracy for attack iterations: {acc_steps.tolist()}')
        print(f'Model: {args.model_name}, adversarial attack success ratio for attack iterations: {adv_succ_ratio_steps.tolist()}')
        print(f'Model: {args.model_name}, adversarial loss for attack iterations: {avg_loss_steps.tolist()}')

    if args.save_results:
        save_path = args.adv_pert_save_path + '/adv_input.pt'
        print("saving adv inputs tensors to path:")
        print(save_path)
        torch.save(x_adv, args.adv_pert_save_path + '/adv_input.pt')
        save_path = args.imgs_save_path + '/adv_inputs'
        print("saving adv inputs images to path:")
        print(save_path)
        save_img_tensors(save_path, x_adv, args.y_test, y_adv, args.labels_str_dict)


if __name__ == '__main__':
    args = get_args()
    run_adv_attacks(args)
