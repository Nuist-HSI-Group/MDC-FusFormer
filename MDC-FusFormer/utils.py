import torch
from torch.autograd import Variable

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda().float()
    return Variable(x, volatile=volatile)

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def batch_ids2words(batch_ids, vocab):

    batch_words = []
    for i in range(batch_ids.size(0)):
        sampled_caption = []
        ids = batch_ids[i,::].cpu().data.numpy()

        for j in range(len(ids)):
            id = ids[j]
            word = vocab.idx2word[id]
            if word == '<end>':
                break
            if '<start>' not in word:
                sampled_caption.append(word)

        for k in sampled_caption:
            if  k==sampled_caption[0]:
                sentence = k     
            else:
                sentence = sentence + ' ' + k

        sentence = u'{}'.format(sentence)   if sampled_caption!=[]  else  u'.'
        batch_words.append(sentence)

    return batch_words

