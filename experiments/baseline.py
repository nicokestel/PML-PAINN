import os

import torch
from torch import nn
from torch.optim import AdamW

import schnetpack.transform as trn

from data_loader import load_data
from nn.mlp import MLP

from schnetpack.datasets.md17 import MD17

def run(molecule='ethanol'):
    work_dir = './mlp_baseline'

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    bs = 10

    # dataset
    md17data = load_data('md17',
                         molecule=molecule,
                         transformations=[
                             trn.ASENeighborList(cutoff=5.),
                             trn.RemoveOffsets(MD17.energy, remove_mean=True, remove_atomrefs=False),
                             trn.CastTo32()
                         ],
                         n_train=950,  # 950
                         n_val=50,  # 50
                         batch_size=bs,
                         work_dir=work_dir)
    
    # dataloader
    train_dl = md17data.train_dataloader()
    val_dl = md17data.val_dataloader()

    # MLP
    n_atoms = md17data.train_dataset[0]['_n_atoms'].item()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mlp = MLP(in_dim=n_atoms * n_atoms,
              hidden_dims=[2048, 2048, 2048],
              out_dim=n_atoms)
    mlp.to(device)
    print(mlp)

    # setup optimizer and training
    optim = AdamW(mlp.parameters())
    epochs = 200
    valid_losses = {'mae': [], 'mse': []}

    # start training
    for epoch in range(epochs):

        # training epoch
        mlp.train()
        for batch_id, b in enumerate(train_dl):
            batch = b['_positions'].view(bs, n_atoms, -1).clone().detach().to(device)
            cdist = torch.cdist(batch, batch).view(bs, -1).to(device)
            #print(cdist.shape)
            pred = mlp(cdist).cpu()
            #print(pred.shape)

            f_mag = torch.linalg.norm(b['forces'].view(bs, n_atoms, 3), dim=-1)
            #print(f_mag.shape)
            loss = nn.functional.l1_loss(pred, f_mag)  # MAE

            optim.zero_grad()
            loss.backward()
            optim.step()

            print('\rEpoch:{} Batch:{}/{} Loss:{:.4f}'.format(epoch + 1,
                                                              batch_id+1,
                                                              len(train_dl),
                                                              loss.item(),
                                                              end=''))
            
        # validation
        mlp.eval()
        valid_mae_loss = 0.0
        valid_mse_loss = 0.0
        best_loss = torch.inf
        for batch_id, b in enumerate(val_dl):
            batch = b['_positions'].view(bs, n_atoms, -1).clone().detach().to(device)
            cdist = torch.cdist(batch, batch).view(bs, -1).to(device)
            pred = mlp(cdist).cpu()

            f_mag = torch.linalg.norm(b['forces'].view(bs, n_atoms, 3), dim=-1)

            mae_loss = nn.functional.l1_loss(pred, f_mag)  # MAE
            mse_loss = nn.functional.mse_loss(pred, f_mag) # MSE
            valid_mae_loss += mae_loss
            valid_mse_loss += mse_loss

        mae_loss /= len(val_dl)
        mse_loss /= len(val_dl)
        valid_losses['mae'].append(mae_loss)
        valid_losses['mse'].append(mse_loss)

        #save model when validation loss is minimum
        if valid_mae_loss < best_loss:
            best_loss = valid_mae_loss
            torch.save(mlp, os.path.join(work_dir, 'mlp_moredata'))
        
        print('Valid MAE Loss:{:.4f}'.format(valid_mae_loss))
        print('Valid MSE Loss:{:.4f}'.format(valid_mse_loss))


