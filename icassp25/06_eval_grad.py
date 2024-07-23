import torch
import os
from pnp_synth.neural import cnn, optimizer
from pnp_synth.neural import loss as losses
import icassp25
import torch.nn.functional as F
import numpy as np
import math
import copy
import sys

opt = sys.argv[1] # sophia / adam
loss_type = sys.argv[2] # ploss / weighted_p
eff_type = sys.argv[3]

data_dir = "/gpfswork/rech/aej/ufg99no/data/ftm_jtfs/" 
#data_dir = "/home/han/localdata/data/ftm_jtfs/"
full_df = icassp25.load_fold(fold="full")
batch_size = 256
scale_factor = 1e-20

nbatch = icassp25.SAMPLES_PER_EPOCH // (10 * batch_size) # however much that covers 10% training set

def evaluate_gradnorm(model, nbatch):
    model.zero_grad()
    # fake forward pass test data through the model 
    model.train()
    for batch_idx, batch_data in enumerate(train_dataset):
        batch_input = batch_data["feature"]
        batch_target = batch_data["y"].cuda()
        # Perform forward pass
        output = model(batch_input).cuda()
        # Compute loss
        if loss_type == "ploss":
            loss = F.mse_loss(output, batch_target)
        elif loss_type == "weighted_p":
            batch_M = batch_data["M"].cuda()
            D = torch.eye(batch_M.shape[1]).double()[None, :, :]
            D = LMA_lambda * D.to("cuda")
            batch_M = batch_M + D
            loss = losses.loss_bilinear(output.double(), batch_target.double(), mu*batch_M)
        # Perform backward pass
        loss.backward()
        if batch_idx + 1 == nbatch: # in the original paper this accounts for 10% of training set
            break
    gradnorm = 0
    for param in model.parameters():
        if param.grad is not None:
            if loss_type == "weighted_p":
                param_norm = (scale_factor * param.grad.data).norm(2)
            elif loss_type == "ploss":
                param_norm = param.grad.data.norm(2)
            gradnorm += param_norm.item() ** 2
    gradnorm = gradnorm ** (1. / 2) / (nbatch)
    if loss_type == "weighted_p":
        gradnorm = gradnorm / scale_factor
    return gradnorm

def get_model_grads(model):
    return [p.grad.data for _, p in model.named_parameters() if \
            hasattr(p, 'grad') and (p.grad is not None)]

def get_model_params(model):
    return [p.data for _, p in model.named_parameters() if \
            hasattr(p, 'grad') and (p.grad is not None)]


def norm_diff(list1, list2=None, scale=False):
    if not list2:
        list2 = [0] * len(list1)
    assert len(list1) == len(list2)
    if scale:
        diff = math.sqrt(sum((scale_factor * (list1[i]-list2[i])).norm()**2 for i in range(len(list1))))
        return diff / scale_factor
    else:
        return math.sqrt(sum((list1[i]-list2[i]).norm()**2 for i in range(len(list1))))

def eval_smooth(prev_model, model, nbatch, num_pts=1):
    alphas = np.arange(1, num_pts+1)/(num_pts+1)
    gnorm = evaluate_gradnorm(prev_model, nbatch)
    update_size = norm_diff(get_model_params(model), \
                                  get_model_params(prev_model)) # norm of weight difference
    max_smooth = -1
    for alpha in alphas:
        new_model = copy.deepcopy(prev_model)
        for n, p in new_model.named_parameters():
            p.data = alpha * p.data + (1-alpha) * {n:p for n, p in model.named_parameters()}[n].data 
            
        evaluate_gradnorm(new_model, nbatch)
        scale = True if loss_type == "weighted_p" else False
        smooth = norm_diff(get_model_grads(new_model), get_model_grads(prev_model), scale=scale)/ (update_size * (1- alpha)) # norm of gradient difference divided by norm of weight difference
        if smooth == np.inf:
            print("smoothness exeeds bounds, why?", update_size)
        max_smooth = max(smooth, max_smooth)
    
    return max_smooth, gnorm


# load data
_, scaler = icassp25.scale_theta(logscale=1)
dataset = cnn.DrumDataModule(
        batch_size=batch_size,
        data_dir=os.path.join(data_dir, "x"),  # path to hdf5 files
        cqt_dir=os.path.join(data_dir, "x"),
        df=full_df,
        weight_dir=os.path.join(data_dir, "M_log"),  # path to gradient folders
        weight_type=None,  # novol, pnp
        feature="cqt",
        logscale=1,
        J=10,
        Q=12,
        sr=22050,
        scaler=scaler,
        num_workers=0,
    )
dataset.setup()
test_dataset = dataset.test_dataloader()
train_dataset = dataset.train_dataloader()


# load models
outdim = 5
#eff_type = "b0"
LMA = {
        'mode': "adaptive", #scheduled / constant
        'accelerator': 0.05,
        'brake': 1,
        'damping': "id"
    }

steps_per_epoch = 100# icassp25.SAMPLES_PER_EPOCH / batch_size
lr = 1e-3
log_interval = 1 # frequency every number of batches to log gradient norm/smoothness 
mu = 1e-20

model = cnn.EffNet(in_channels=1, outdim=outdim, loss=loss_type, eff_type=eff_type, 
                       scaler=scaler, LMA=LMA, steps_per_epoch=steps_per_epoch,
                         var=0.5, save_path="./ftm", lr=lr, minmax=1, 
                         logtheta=1, opt=opt, mu=mu)


model = model.cuda()



# initialize optimizer

if opt == "adam":
    optimizer_curr = torch.optim.Adam(model.parameters(), lr=model.lr)
elif opt == "sophia":
    optimizer_curr = optimizer.SophiaG(params=model.parameters(), lr=model.lr, betas=(0.965, 0.99), rho = 0.01, weight_decay=1e-1)

# forward pass

print("start training")
smooths = []
gradnorms = []
for batch_idx, batch_data in enumerate(train_dataset): # see once all the training set
    prev_model = copy.deepcopy(model)
    print("step {}:".format(batch_idx))
    model.train()
    batch_input = batch_data["feature"]
    batch_target = batch_data["y"].to("cuda")
    batch_M = batch_data["M"].cuda()
    if batch_idx == 0:
        LMA_lambda = batch_data['lambda0'].to("cuda")
    elif batch_idx == steps_per_epoch // 2: # nonstationary objective
        LMA_lambda = batch_data['lambda0'].to("cuda") * LMA["accelerator"] #(decay the damping coefficients)
    optimizer_curr.zero_grad()  # Clear existing gradients
    # Perform forward pass
    output = model(batch_input).cuda()
    # Compute loss
    if loss_type == "ploss":
        loss = F.mse_loss(output, batch_target)
    elif loss_type == "weighted_p":
        batch_M = batch_data["M"].cuda()
        D = torch.eye(batch_M.shape[1]).double()[None, :, :]
        D = LMA_lambda * D.to("cuda")
        batch_M = batch_M + D
        loss = losses.loss_bilinear(output.double(), batch_target.double(), mu * batch_M)
    # Perform backward pass
    loss.backward()
    optimizer_curr.step() # one step of weight update
    
    if batch_idx % log_interval == 0: 
        smoothness, gradnorm = eval_smooth(prev_model, model, nbatch)
        smooths.append(smoothness)
        gradnorms.append(gradnorm)
        print("iter {}, gradient norm {}, smoothness {} ".format(batch_idx, gradnorm, smoothness))

    if batch_idx > steps_per_epoch:
        break # break after seeing the entire training set
    
np.save("./{}_{}_{}_gradnorms.npy".format(eff_type, opt, loss_type), gradnorms)
np.save("./{}_{}_{}_smoothness.npy".format(eff_type, opt, loss_type), smooths)
