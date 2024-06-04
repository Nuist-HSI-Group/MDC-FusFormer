import time

import torch.optim
from models.MDC import MDC
from utils import *
from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam, calc_mssim
from data_loader import build_datasets
import args_parser
import cv2
from time import *
from thop import profile

args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(args)


def main():
    if args.dataset == 'PaviaU':
        args.n_bands = 103
    elif args.dataset == 'Pavia':
        args.n_bands = 102
    elif args.dataset == 'Washington':
        args.n_bands = 191
    elif args.dataset == 'Houston_HSI':
        args.n_bands = 144
    elif args.dataset == 'Salinas_corrected':
        args.n_bands = 204
    # Custom dataloader
    train_list, test_list = build_datasets(args.root,
                                           args.dataset,
                                           args.image_size,
                                           args.n_select_bands,
                                           args.scale_ratio)
    # Build the models
    if args.arch == 'MDC':
        model = MDC(args.arch,
                    args.scale_ratio,
                    args.n_select_bands,
                    args.n_bands,
                    args.dataset)
    # Load the trained model parameters
    model_path = args.model_path.replace('dataset', args.dataset) \
        .replace('arch', args.arch)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
        print('Load the chekpoint of {}'.format(model_path))

    test_ref, test_lr, test_hr = test_list

    model.eval()

    # Set mini-batch dataset
    ref = test_ref.float().detach()
    lr = test_lr.float().detach()
    hr = test_hr.float().detach()

    begin_time = time()
    out = model(lr, hr)
    end_time = time()
    run_time = (end_time - begin_time) * 1000

    # flops
    print()
    flops, params = profile(model, inputs=(lr, hr,))
    average_times = 1000
    print('Dataset:   {}'.format(args.dataset))
    print('Arch:   {}'.format(args.arch))
    print('params(M),', params / 1e6)
    print('flops(G),', flops / 1e9)
    model.cuda()
    lr = lr.cuda()
    hr = hr.cuda()
    a_time = time()
    for i in range(average_times):
        out= model(lr, hr)
    b_time = time()
    print('avarage times', average_times)
    print('test time(ms)', 1000 * (b_time - a_time) / average_times)
    print()

    print()
    print()
    print('Dataset:   {}'.format(args.dataset))
    print('Arch:   {}'.format(args.arch))
    print()

    ref = ref.detach().cpu().numpy()
    out = out.detach().cpu().numpy()

    psnr = calc_psnr(ref, out)
    rmse = calc_rmse(ref, out)
    ergas = calc_ergas(ref, out)
    sam = calc_sam(ref, out)
    ssim = calc_mssim(ref, out)

    print('RMSE:   {:.4f};'.format(rmse))
    print('PSNR:   {:.4f};'.format(psnr))
    print('ERGAS:   {:.4f};'.format(ergas))
    print('SAM:   {:.4f}.'.format(sam))
    print('SSIM:   {:.4f}.'.format(ssim))

    # bands order
    if args.dataset == 'PaviaU' or args.dataset == 'Pavia':
        red = 67
        green = 29
        blue = 1
    elif args.dataset == 'Washington':
        red = 54
        green = 34
        blue = 10
    elif args.dataset == 'Houston_HSI':
        red = 28
        green = 14
        blue = 3
    elif args.dataset == 'Salinas_corrected':
        red = 28
        green = 14
        blue = 3

    lr = np.squeeze(test_lr.detach().cpu().numpy())
    lr_red = lr[red, :, :][:, :, np.newaxis]
    lr_green = lr[green, :, :][:, :, np.newaxis]
    lr_blue = lr[blue, :, :][:, :, np.newaxis]
    lr = np.concatenate((lr_blue, lr_green, lr_red), axis=2)
    lr = 255 * (lr - np.min(lr)) / (np.max(lr) - np.min(lr))
    lr = cv2.resize(lr, (out.shape[2], out.shape[3]), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('./figs/{}_lr.jpg'.format(args.dataset), lr)

    out = np.squeeze(out)
    out_red = out[red, :, :][:, :, np.newaxis]
    out_green = out[green, :, :][:, :, np.newaxis]
    out_blue = out[blue, :, :][:, :, np.newaxis]
    out = np.concatenate((out_blue, out_green, out_red), axis=2)
    out = 255 * (out - np.min(out)) / (np.max(out) - np.min(out))
    cv2.imwrite('./figs/{}_{}_out.jpg'.format(args.dataset, args.arch), out)

    ref = np.squeeze(ref)
    ref_red = ref[red, :, :][:, :, np.newaxis]
    ref_green = ref[green, :, :][:, :, np.newaxis]
    ref_blue = ref[blue, :, :][:, :, np.newaxis]
    ref = np.concatenate((ref_blue, ref_green, ref_red), axis=2)
    ref = 255 * (ref - np.min(ref)) / (np.max(ref) - np.min(ref))
    cv2.imwrite('./figs/{}_ref.jpg'.format(args.dataset), ref)

    lr_dif = np.uint8(1.5 * np.abs((lr - ref)))
    lr_dif = cv2.cvtColor(lr_dif, cv2.COLOR_BGR2GRAY)
    lr_dif = cv2.applyColorMap(lr_dif, cv2.COLORMAP_JET)
    cv2.imwrite('./figs/{}_lr_dif.jpg'.format(args.dataset), lr_dif)

    out_dif = np.uint8(1.5 * np.abs((out - ref)))
    out_dif = cv2.cvtColor(out_dif, cv2.COLOR_BGR2GRAY)
    out_dif = cv2.applyColorMap(out_dif, cv2.COLORMAP_JET)
    cv2.imwrite('./figs/{}_{}_out_dif.jpg'.format(args.dataset, args.arch), out_dif)

    ref_dif = np.uint8(1.5 * np.abs((ref - ref)))
    ref_dif = cv2.cvtColor(ref_dif, cv2.COLOR_BGR2GRAY)
    ref_dif = cv2.applyColorMap(ref_dif, cv2.COLORMAP_JET)
    cv2.imwrite('./figs/{}_{}_ref_dif.jpg'.format(args.dataset, args.arch), ref_dif)


if __name__ == '__main__':
    main()
