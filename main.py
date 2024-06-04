import torch.optim
from models.MDC import MDC
from utils import *
from data_loader import build_datasets
from validate import validate
from train import train
import args_parser

args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print (args)

def main():

    train_list, test_list = build_datasets(args.root,
                                           args.dataset,
                                           args.image_size,
                                           args.n_select_bands,
                                           args.scale_ratio)
    if args.dataset == 'PaviaU':
      args.n_bands = 103
    elif args.dataset == 'Pavia':
      args.n_bands = 102
    elif args.dataset == 'Washington':
      args.n_bands = 191
    elif args.dataset == 'Salinas_corrected':
      args.n_bands = 204
    elif args.dataset == 'Houston_HSI':
      args.n_bands = 144
    # Build the models
    model = MDC(args.arch,
                args.scale_ratio,
                args.n_select_bands,
                args.n_bands,
                args.dataset).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    parameter_nums = sum(p.numel() for p in model.parameters())
    print("Model size:", str(float(parameter_nums / 1e6)) + 'M')

    model_path = args.model_path.replace('dataset', args.dataset) \
                                .replace('arch', args.arch)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
        print ('Load the chekpoint of {}'.format(model_path))
        recent_psnr = validate(test_list,
                                args.arch,
                                model,
                                0,
                                args.n_epochs)
        print ('psnr: ', recent_psnr)

    criterion = nn.MSELoss().cuda()

    best_psnr = 0
    best_psnr = validate(test_list,
                          args.arch,
                          model,
                          0,
                          args.n_epochs)
    print ('the previous psnr: ', best_psnr)

    # Epochs
    print ('Start Training: ')
    best_epoch = 0
    for epoch in range(args.n_epochs):
        print ('Train_Epoch_{}: '.format(epoch))
        train(train_list,
              args.image_size,
              args.scale_ratio,
              args.n_bands,
              args.arch,
              model,
              optimizer,
              criterion,
              epoch,
              args.n_epochs)

        print ('Val_Epoch_{}: '.format(epoch))
        recent_psnr = validate(test_list,
                                args.arch,
                                model,
                                epoch,
                                args.n_epochs)
        print ('psnr: ', recent_psnr)

        is_best = recent_psnr > best_psnr
        best_psnr = max(recent_psnr, best_psnr)
        if is_best:
          best_epoch=epoch
          if best_psnr > 0:
            torch.save(model.state_dict(), model_path)
            print ('Saved!')
            print ('')
        print('best psnr:', best_psnr, 'at epoch:', best_epoch)

    print ('best_psnr: ', best_psnr)

if __name__ == '__main__':
    main()
