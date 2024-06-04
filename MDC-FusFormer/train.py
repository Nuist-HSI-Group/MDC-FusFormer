from utils import to_var, batch_ids2words
import random
import torch.nn.functional as F

def train(train_list,
          image_size,
          scale_ratio,
          n_bands,
          arch,
          model,
          optimizer,
          criterion,
          epoch,
          n_epochs):
    train_ref, train_lr, train_hr = train_list

    h, w = train_ref.size(2), train_ref.size(3)
    h_str = random.randint(0, h-image_size-1)
    w_str = random.randint(0, w-image_size-1)

    train_ref = train_ref[:, :, h_str:h_str+image_size, w_str:w_str+image_size]
    train_lr = F.interpolate(train_ref, scale_factor=1/(scale_ratio*1.0))
    train_hr = train_hr[:, :, h_str:h_str+image_size, w_str:w_str+image_size]

    model.train()

    image_lr = to_var(train_lr).detach()
    image_hr = to_var(train_hr).detach()
    image_ref = to_var(train_ref).detach()

    optimizer.zero_grad()

    out = model(image_lr, image_hr)

    if 'RNET' in arch:
        loss_fus = criterion(out, image_ref)
        loss = loss_fus
    else:
        loss = criterion(out, image_ref)

    loss.backward()
    optimizer.step()

    print('Epoch [%d/%d], Loss: %.4f'
          %(epoch,
            n_epochs,
            loss,
            )
         )
