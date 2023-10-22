import torch


def cxcywh_to_gcxgcy(cxcywh, default_cxcywh):
    return torch.cat([(cxcywh[:, :2] - default_cxcywh[:, :2]) / (default_cxcywh[:, 2:] / 10),
                      torch.log(cxcywh[:, 2:] / default_cxcywh[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcywh(gcxgcywh, default_cxcywh):
    return torch.cat([gcxgcywh[:, :2] * default_cxcywh[:, 2:] / 10 + default_cxcywh[:, :2],
                      torch.exp(gcxgcywh[:, 2:] / 5) * default_cxcywh[:, 2:]], 1)


def save_checkpoint(epoch, model, optimizer):
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint_ssd300.pth.tar'
    torch.save(state, filename)
    
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)