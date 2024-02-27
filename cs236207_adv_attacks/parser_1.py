import torch
import torch.backends.cudnn as cudnn
import argparse
from os import mkdir
from os.path import isdir
from torchvision.utils import save_image
from robustbench.data import load_cifar10
from robustbench.utils import load_model
import torchvision.transforms as transforms
from models.cifar10.resnet import ResNet18

from attacks.pgd_attacks import PGD, UPGD


def parse_args():
    parser = argparse.ArgumentParser()
    # run args
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: random)')
    parser.add_argument('--gpus', default='0', help='List of GPUs used - e.g 0,1,3')
    parser.add_argument('--force_cpu', action='store_true', help='Force pytorch to run in CPU mode.')
    parser.add_argument('--attack_verbose', action='store_true')
    parser.add_argument('--runner_verbose', action='store_true')
    # data args
    parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, cifar100, imagenet')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--n_examples', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--report_info', action='store_true', help='additional info and non final results will be reported as well')
    parser.add_argument('--save_results', action='store_true',
                        help='save the produced results')
    # model args
    parser.add_argument('--model_name', type=str, default='', help='model name to load from robustness (default: use pretrained ResNet18 model)')
    #parser.add_argument('--model_name', type=str, default='Wong2020Fast', help='model name to load from robustness (default: use pretrained ResNet18 model)')
    # parser.add_argument('--model_name', type=str, default='student', help='model name to load from robustness (default: use pretrained ResNet18 model)')

    # Use --model_name Wong2020Fast for robust PreActResNet-18 model
    # attack args
    # pgd attacks args
    parser.add_argument('--attack', type=str, default='UPGD', help='PGD, UPGD')
    parser.add_argument('--eps_l_inf_from_255', type=int, default=8)
    parser.add_argument('--n_iter', type=int, default=30)
    parser.add_argument('--n_restarts', type=int, default=1, help='number of restart iterations for pgd_attacks')
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--att_init_zeros', action='store_true', help='initialize the adversarial pertubation to zeroes (default: random initialization)')

    args = parser.parse_args()
    print("args")
    print(args)
    return args


def compute_run_args(args):
    if args.gpus is not None and not args.force_cpu and torch.cuda.is_available():
        args.gpus = [int(i) for i in args.gpus.split(',')]
        cudnn.enabled = True
        cudnn.benchmark = True
        args.device = torch.device('cuda:1')
        torch.cuda.manual_seed(args.seed)
    else:
        args.gpus = []
        args.device = torch.device('cpu')
    torch.cuda.set_device(args.device)
    torch.cuda.init()
    print('Running inference on device \"{}\"'.format(args.device))
    return args


def compute_data_args(args):
    if args.n_examples < args.batch_size:
        args.batch_size = args.n_examples
    args.data_RGB_start = [0, 0, 0]
    args.data_RGB_end = [1, 1, 1]
    args.data_RGB_offset = [0, 0, 0]
    args.data_RGB_size = [1, 1, 1]

    #load cifar10/100
    args.labels_str_dict = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
                            'truck']
    transforms_list = [transforms.ToTensor()]
    transforms_test = transforms.Compose(transforms_list)
    args.n_classes = 10
    args.x_test, args.y_test = load_cifar10(n_examples=args.n_examples,
                                            data_dir=args.data_dir,
                                            transforms_test=transforms_test)
    args.n_examples = args.x_test.shape[0]
    args.data_channels = args.x_test.shape[1]
    args.data_shape = list(args.x_test.shape)[1:]
    args.data_pixels = args.data_shape[1] * args.data_shape[2]
    args.dtype = args.x_test.dtype
    return args


def compute_models_args(args):
    if args.model_name == 'Wong2020Fast':
        args.model = load_model(args.model_name, dataset=args.dataset, threat_model='Linf').to(args.device)
    elif args.model_name == 'student':
        args.model = load_model('Standard', dataset=args.dataset, threat_model='Linf').to(args.device)
        state_dict = torch.load('models/cifar10/Linf/student_model_best_699.pt', map_location=args.device)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        args.model.load_state_dict(new_state_dict)
    else:
        #cifar 10
        args.model_name = 'ResNet18'
        args.model = ResNet18(args.device)
        args.model.load_state_dict(torch.load('models/cifar10/resnet18.pt', map_location=args.device))
    args.model.eval()

    return args


def compute_attack_args(args):
    args.criterion = torch.nn.CrossEntropyLoss(reduction='none')
    args.eps_l_inf = args.eps_l_inf_from_255 / 255
    args.att_rand_init = not args.att_init_zeros

    attack_dict = {'PGD': PGD, 'UPGD': UPGD}
    args.attack_name = args.attack
    if args.attack in attack_dict:
        args.attack = attack_dict[args.attack]
    else:
        args.attack_name = 'PGD'
        args.attack = PGD
    print("Testing models under " + args.attack_name + " attack")

    args.att_misc_args = {'device': args.device,
                   'dtype': args.dtype,
                   'batch_size': args.batch_size,
                   'data_shape': args.data_shape,
                   'data_RGB_start': args.data_RGB_start,
                   'data_RGB_end': args.data_RGB_end,
                   'data_RGB_size': args.data_RGB_size,
                   'verbose': args.attack_verbose,
                   'report_info': args.report_info}

    args.att_pgd_args = {'norm': 'Linf',
                   'eps': args.eps_l_inf,
                   'n_restarts': args.n_restarts,
                   'n_iter': args.n_iter,
                   'alpha': args.alpha,
                   'rand_init': args.att_rand_init}

    args.attack_obj_str = "norm_Linf_eps_from_255_" + str(args.eps_l_inf_from_255) + \
                          "_iter_" + str(args.n_iter) + \
                          "_restarts_" + str(args.n_restarts) + \
                          "_alpha_" + str(args.alpha).replace('.', '_') + \
                          "_rand_init_" + str(args.att_rand_init)

    if attack_dict[args.attack_name] == PGD:
        args.attack_obj = args.attack(args.model, args.criterion,
                                        misc_args=args.att_misc_args,
                                        pgd_args=args.att_pgd_args)
    elif attack_dict[args.attack_name] == UPGD:
        args.attack_obj = args.attack(args.model, args.criterion,
                                misc_args=args.att_misc_args,
                                pgd_args=args.att_pgd_args,
                                x_orig=args.x_test, y_orig=args.y_test,
                                model_name = args.model_name)

    return args


def compute_save_path_args(args):
    if args.results_dir is None or not len(args.results_dir) or not args.save_results:
        return args
    if not isdir(args.results_dir):
        mkdir(args.results_dir)
    args.data_save_path = args.results_dir + '/' + args.dataset
    if not isdir(args.data_save_path):
        mkdir(args.data_save_path)
    args.data_save_path = args.data_save_path + '/n_examples_' + str(args.n_examples)
    if not isdir(args.data_save_path):
        mkdir(args.data_save_path)
    args.model_save_path = args.data_save_path + '/model_' + args.model_name
    if not isdir(args.model_save_path):
        mkdir(args.model_save_path)
    args.attack_class_save_path = args.model_save_path + '/attack_' + args.attack_name
    if not isdir(args.attack_class_save_path):
        mkdir(args.attack_class_save_path)
    args.attack_obj_save_path = args.attack_class_save_path + '/' + args.attack_obj_str
    if not isdir(args.attack_obj_save_path):
        mkdir(args.attack_obj_save_path)
    args.results_save_path = args.attack_obj_save_path
    args.adv_pert_save_path = args.results_save_path + '/perturbations'
    if not isdir(args.adv_pert_save_path):
        mkdir(args.adv_pert_save_path)
    args.imgs_save_path = args.results_save_path + '/images'
    if not isdir(args.imgs_save_path):
        mkdir(args.imgs_save_path)
    if args.save_results:
        print("Results save path:")
        print(args.results_save_path)
    return args


def get_args():
    args = parse_args()
    args = compute_run_args(args)
    args = compute_data_args(args)
    args = compute_models_args(args)
    args = compute_attack_args(args)
    args = compute_save_path_args(args)
    return args


def save_img_tensors(path, x, gt=None, pred=None, labels_str_dict=None, save_type=".pdf"):
    if not isdir(path):
        mkdir(path)
    for idx, input in enumerate(x):
        save_path = path + '/' + str(idx)
        if not isdir(save_path):
            mkdir(save_path)
        name = "img"
        if labels_str_dict is not None:
            if gt is not None:
                input_gt = gt[idx]
                gt_label = labels_str_dict[input_gt]
                name += "_gt_label_" + gt_label
            if pred is not None:
                input_pred = pred[idx]
                pred_label = labels_str_dict[input_pred]
                name += "_pred_label_" + pred_label
        name += save_type
        save_image(input, save_path + '/' + name)
