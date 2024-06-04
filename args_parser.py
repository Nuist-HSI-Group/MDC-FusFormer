import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-arch', type=str, default='MDC')

    parser.add_argument('-root', type=str, default='./data')
    parser.add_argument('-dataset',type=str, default='Houston_HSI', choices=['PaviaU', 'Pavia', 'Washington','Salinas_corrected','Houston_HSI'])
    parser.add_argument('--scale_ratio', type=float, default=4)
    parser.add_argument('--n_bands', type=int, default=0)
    parser.add_argument('--n_select_bands', type=int, default=5)
    parser.add_argument('--model_path', type=str,
                        default='./checkpoints/dataset_arch.pkl',
                        help='path for trained encoder')
    parser.add_argument('--train_dir', type=str, default='./data/dataset/train',
                        help='directory for resized images')
    parser.add_argument('--val_dir', type=str, default='./data/dataset/val',
                        help='directory for resized images')


    parser.add_argument('--n_epochs', type=int, default=10000,
                        help='end epoch for training')

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--image_size', type=int, default=128)

    args = parser.parse_args()
    return args
