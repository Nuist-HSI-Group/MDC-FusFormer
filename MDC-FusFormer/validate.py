from utils import *
from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam, calc_mssim

def validate(test_list, arch, model, epoch, n_epochs):
    test_ref, test_lr, test_hr = test_list
    model.eval()

    with torch.no_grad():

        ref = to_var(test_ref).detach()
        lr = to_var(test_lr).detach()
        hr = to_var(test_hr).detach()
        out = model(lr, hr)

        ref = ref.detach().cpu().numpy()
        out = out.detach().cpu().numpy()

        rmse = calc_rmse(ref, out)
        psnr = calc_psnr(ref, out)
        ergas = calc_ergas(ref, out)
        sam = calc_sam(ref, out)
        SSIM = calc_mssim(ref, out)

        with open('MDC.txt', 'a') as f:
            f.write(str(epoch) + ',' + str(rmse) + ',' + str(psnr) + ',' + str(ergas) + ',' + str(sam) + ',' + ',' + str(SSIM) + ',' +  '\n')

    return psnr