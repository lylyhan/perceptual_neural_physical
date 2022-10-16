import torch
from pnp_synth.neural import forward
import torch.nn as nn
from pnp_synth import utils


def loss_spec(outputs, y, specloss,scaler):
    #put through synth ##TODO: need the batch processing!!! or make a loop
    #temporary loop
    wav_gt = []
    wav_pred = []
    for i in range(y.shape[0]):
        wav_gt.append(forward.pnp_forward(y[i,:], Phi=nn.Identity(), g=utils.x_from_theta, scaler=scaler))
        wav_pred.append(forward.pnp_forward(outputs[i,:], Phi=nn.Identity(), g=utils.x_from_theta, scaler=scaler))
    wav_gt = torch.stack(wav_gt)
    wav_pred = torch.stack(wav_pred)
    loss = specloss(wav_pred, wav_gt)
    return loss

def loss_bilinear(outputs, y, M, l):
    diff = outputs - y 
    M = M + l * torch.eye(M.shape[1]).cuda()
    loss = torch.bmm(torch.bmm(diff[:,None,:], M), diff[:,:,None])
    loss = torch.relu(loss) #/1e+5
    return 0.5*torch.mean(loss.squeeze())
