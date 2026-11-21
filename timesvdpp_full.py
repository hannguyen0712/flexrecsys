import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data import set_seed
from metrics import rmse


def setup_data(folder):
    folder = Path(folder)
    return setup_frames(pd.read_parquet(folder / 'train.parquet'),
                        pd.read_parquet(folder / 'val.parquet'))


def setup_frames(train, val):
    users = np.sort(train.userId.unique())
    items = np.sort(train.movieId.unique())
    user_map = {int(x): i for i, x in enumerate(users)}
    item_map = {int(x): i for i, x in enumerate(items)}
    def encode(frame):
        u = frame.userId.map(user_map).fillna(-1).to_numpy(np.int64)
        i = frame.movieId.map(item_map).fillna(-1).to_numpy(np.int64)
        d = (frame.timestamp.to_numpy(np.int64) // 86400).astype(np.int64)
        r = frame.rating.to_numpy(np.float32)
        return u, i, d, r
    tu, ti, td, tr = encode(train)
    vu, vi, vd, vr = encode(val)
    if np.any(tu < 0) or np.any(ti < 0):
        raise RuntimeError('Unexpected unmapped training ID')
    if np.any(vu < 0):
        raise RuntimeError('Validation includes new users: needs cold-user handling')
    avg_day = np.bincount(tu, weights=td, minlength=len(users)) / np.bincount(tu, minlength=len(users))
    first_day, last_day = int(min(td.min(), vd.min())), int(max(td.max(), vd.max()))
    n_bins = 30
    def extras(u, d):
        diff = d - avg_day[u]
        dev = (np.sign(diff) * (np.abs(diff) / 365.0) ** .4).astype(np.float32)
        day_index = (d - first_day).astype(np.int64)
        bins = np.clip(((d - first_day) * n_bins // (last_day - first_day + 1)), 0, n_bins-1).astype(np.int64)
        return dev, day_index, bins
    tdev, tday, tbin = extras(tu, td)
    vdev, vday, vbin = extras(vu, vd)
    # A user-day lookup includes only days observed in training. Validation days
    # absent from training have user-day bias exactly zero, not a learned future value.
    keys = {(int(u),int(d)):idx+1 for idx, (u,d) in enumerate(sorted(set(zip(tu,td))))}
    t_day_key = np.fromiter((keys[(int(u),int(d))] for u,d in zip(tu,td)), dtype=np.int64, count=len(tu))
    v_day_key = np.fromiter((keys.get((int(u),int(d)),0) for u,d in zip(vu,vd)), dtype=np.int64, count=len(vu))
    # Sorted by user, with offsets describing each user's training-rated movies.
    order = np.argsort(tu, kind='stable')
    history_items = ti[order]
    hist_counts = np.bincount(tu, minlength=len(users))
    hist_offsets = np.r_[0, np.cumsum(hist_counts)].astype(np.int64)
    return dict(train=train,val=val,users=users,items=items,
                t=(tu,ti,tdev,tbin,t_day_key,tr),
                v=(vu,vi,vdev,vbin,v_day_key,vr),
                history_items=history_items,hist_offsets=hist_offsets,
                n_bins=n_bins,n_day_keys=len(keys)+1)


class TimeSVDpp(nn.Module):
    def __init__(self, n_users, n_items, n_bins, n_day_keys, hist_items, hist_offsets,
                 mean, factors=32, ablate=()):
        super().__init__()
        self.ablate = frozenset(ablate)
        self.p = nn.Embedding(n_users, factors)
        self.q = nn.Embedding(n_items, factors)
        self.y = nn.Embedding(n_items, factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.item_bin_bias = nn.Embedding(n_items*n_bins, 1)
        self.user_alpha = nn.Embedding(n_users, 1)
        self.user_day_bias = nn.Embedding(n_day_keys, 1, padding_idx=0)
        self.user_factor_alpha = nn.Embedding(n_users, factors)
        self.user_day_factor = nn.Embedding(n_day_keys, factors, padding_idx=0)
        self.n_bins = n_bins
        self.register_buffer('global_mean',torch.tensor(float(mean)))
        self.register_buffer('hist_items',torch.as_tensor(hist_items,dtype=torch.long))
        self.register_buffer('hist_offsets',torch.as_tensor(hist_offsets,dtype=torch.long))
        counts = self.hist_offsets[1:] - self.hist_offsets[:-1]
        self.register_buffer('hist_scale', counts.clamp(min=1).float().rsqrt())
        for layer in (self.p,self.q,self.y):
            nn.init.normal_(layer.weight,mean=0,std=0.025)
        for layer in (self.user_bias,self.item_bias,self.item_bin_bias,
                      self.user_alpha,self.user_day_bias,self.user_factor_alpha,
                      self.user_day_factor):
            nn.init.zeros_(layer.weight)

    def implicit_all(self):
        # Once per batch: sum feedback vectors for users appearing in the batch.
        # For CPU efficiency, gather each user's history via embedding_bag.
        # Called as a full-user operation and indexed afterwards: gradients still
        # flow through y and user histories remain training-only.
        if 'implicit' in self.ablate:
            return None
        return torch.nn.functional.embedding_bag(
            self.hist_items, self.y.weight, self.hist_offsets[:-1], mode='sum'
        ) * self.hist_scale.unsqueeze(1)

    def forward(self, u, i, dev, bins, day_keys, implicit):
        a = self.ablate
        user_vector = self.p(u)
        if 'user_drift' not in a:
            user_vector = user_vector + self.user_factor_alpha(u)*dev.unsqueeze(1)
        if 'user_day' not in a:
            user_vector = user_vector + self.user_day_factor(day_keys)
        if 'implicit' not in a:
            user_vector = user_vector + implicit[u]
        out = (self.fallback(u, dev, day_keys) + self.item_bias(i).squeeze(1)
               + (self.q(i)*user_vector).sum(dim=1))
        if 'item_bin' not in a:
            out = out + self.item_bin_bias(i*self.n_bins+bins).squeeze(1)
        return out

    def fallback(self, u, dev, day_keys):
        a = self.ablate
        out = self.global_mean + self.user_bias(u).squeeze(1)
        if 'user_drift' not in a:
            out = out + self.user_alpha(u).squeeze(1)*dev
        if 'user_day' not in a:
            out = out + self.user_day_bias(day_keys).squeeze(1)
        return out


def evaluate(model, v, device, batch_size=8192, rating_scale=(1, 5), return_pred=False):
    u,i,dev,b,day,r = v
    pred = np.empty(len(r),dtype=np.float32)
    model.eval()
    with torch.no_grad():
        implicit = model.implicit_all()
        for lo in range(0,len(r),batch_size):
            hi=min(lo+batch_size,len(r))
            U=torch.tensor(u[lo:hi],dtype=torch.long,device=device)
            D=torch.tensor(dev[lo:hi],dtype=torch.float32,device=device)
            K=torch.tensor(day[lo:hi],dtype=torch.long,device=device)
            out=model.fallback(U,D,K)
            mask=i[lo:hi]>=0
            if mask.any():
                ix=torch.tensor(np.nonzero(mask)[0],dtype=torch.long,device=device)
                I=torch.tensor(i[lo:hi][mask],dtype=torch.long,device=device)
                B=torch.tensor(b[lo:hi][mask],dtype=torch.long,device=device)
                out[ix]=model(U[ix],I,D[ix],B,K[ix],implicit)
            pred[lo:hi]=out.clamp(*rating_scale).cpu().numpy()
    return (rmse(r,pred), pred) if return_pred else rmse(r,pred)


def fit_timesvdpp(d, factors=32, epochs=8, batch_size=4096, lr=.003, weight_decay=.0001,
                  device='cpu', rating_scale=(1, 5), checkpoint=None, verbose=True, ablate=(),
                  select=True):
    # trains on setup_data/setup_frames output, keeps the best-validation-epoch weights in memory.
    # select=False keeps the last epoch instead, for when d['v'] is the test split
    device=torch.device(device)
    tu,ti,tdev,tbins,tkeys,tr=d['t']
    model=TimeSVDpp(len(d['users']),len(d['items']),d['n_bins'],d['n_day_keys'],
                    d['history_items'],d['hist_offsets'],float(tr.mean()),factors,ablate).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    ds=TensorDataset(*[
        torch.tensor(x,dtype=(torch.float32 if idx in [2,5] else torch.long))
        for idx,x in enumerate(d['t'])
    ])
    dl=DataLoader(ds,batch_size=batch_size,shuffle=True)
    best=float('inf'); best_epoch=None; best_state=None; history=[]
    for epoch in range(1,epochs+1):
        model.train(); total=0.
        # Computing implicit_all separately each minibatch is required for autograd after updates
        # CPU-expensive reference
        for U,I,D,B,K,R in dl:
            U,I,D,B,K,R=[z.to(device) for z in (U,I,D,B,K,R)]
            optimizer.zero_grad()
            implicit=model.implicit_all()
            prediction=model(U,I,D,B,K,implicit)
            loss=((prediction-R)**2).mean()
            loss.backward()
            optimizer.step()
            total+=((prediction.detach()-R)**2).sum().item()
        score=evaluate(model,d['v'],device,rating_scale=rating_scale)
        history.append({'epoch':epoch,'train_rmse':float(np.sqrt(total/len(tr))),'val_rmse':score})
        if verbose:
            print(f'Epoch {epoch:02d} | Train RMSE {np.sqrt(total/len(tr)):.5f} | '
                  f'Validation RMSE {score:.5f}',flush=True)
        if not select:
            best,best_epoch=score,epoch
        elif score<best:
            best,best_epoch=score,epoch
            best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            if checkpoint:
                Path(checkpoint).parent.mkdir(exist_ok=True)
                torch.save(model.state_dict(),checkpoint)
    if select:
        model.load_state_dict(best_state)
    return model, best, best_epoch, history


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/netflix_day_dev')
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--factors',type=int,default=32)
    p.add_argument('--batch-size',type=int,default=4096)
    p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--weight-decay',type=float,default=.0001)
    p.add_argument('--seed',type=int,default=575)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    set_seed(args.seed)
    d=setup_data(Path(args.data))
    tr=d['t'][-1]
    print(f"Train: {len(tr):,} | Validation: {len(d['v'][-1]):,} | "
          f"Users: {len(d['users']):,} | Movies: {len(d['items']):,}",flush=True)
    print(f"Unseen validation movies: {(d['v'][1]<0).sum():,}",flush=True)
    start=time.time()
    _,best,best_epoch,_=fit_timesvdpp(d,args.factors,args.epochs,args.batch_size,args.lr,
                                      args.weight_decay,args.device,
                                      checkpoint='results/timesvdpp_netflix_best.pt')
    output=dict(model='experimental_TimeSVD++',dataset=args.data,seed=args.seed,
                factors=args.factors,epochs=args.epochs,lr=args.lr,
                weight_decay=args.weight_decay,batch_size=args.batch_size,
                best_epoch=best_epoch,best_validation_rmse=best,
                elapsed_seconds=round(time.time()-start,1),
                train_ratings=len(tr),validation_ratings=len(d['v'][-1]),
                unseen_validation_movies=int((d['v'][1]<0).sum()),
                notes='Offline training-only implicit histories; ranking metrics come from '
                      'evaluate_timesvdpp_ranking.py')
    Path('results/timesvdpp_netflix_validation.json').write_text(json.dumps(output,indent=2))
    print('Saved results/timesvdpp_netflix_validation.json',flush=True)


if __name__=='__main__':
    main()