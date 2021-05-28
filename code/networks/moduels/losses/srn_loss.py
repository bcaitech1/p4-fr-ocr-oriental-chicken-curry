import torch
import torch.nn.functional as F

def cal_loss(pred, gold, PAD, smoothing='1'):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing=='0':
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(0)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    elif smoothing == '1':
        loss = F.cross_entropy(pred, gold, ignore_index=PAD)
    else:
        # loss = F.cross_entropy(pred, gold, ignore_index=PAD)
        loss = F.cross_entropy(pred, gold)

    return loss


def cal_performance(preds, gold, PAD, smoothing='1'):
    ''' Apply label smoothing if needed '''

    loss = 0.
    n_correct = 0
    weights = [1.0, 0.15, 2.0]
    for ori_pred, weight in zip(preds, weights):
        pred = ori_pred.view(-1, ori_pred.shape[-1])
        # debug show
        t_gold = gold.view(ori_pred.shape[0], -1)
        t_pred_index = ori_pred.max(2)[1]

        tloss = cal_loss(pred, gold, PAD, smoothing=smoothing)
        if torch.isnan(tloss):
            print('have nan loss')
            continue
        else:
            loss += tloss * weight

        pred = pred.max(1)[1]
        gold = gold.contiguous().view(-1)
        n_correct = pred.eq(gold)
        non_pad_mask = gold.ne(PAD)
        n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct