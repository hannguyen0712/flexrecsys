import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from data import set_seed
from metrics import evaluate_ranking
from timesvdpp_full import TimeSVDpp, setup_data


def recommend_timesvdpp(model, data, users, k=20, batch_size=8192, device='cpu',
                        rating_scale=(1, 5), verbose=True):
    # One recommendation snapshot per user, at their first validation date.
    # Validation labels are used only by metrics.evaluate_ranking afterwards.
    device = torch.device(device)
    val = data['val']
    _, _, v_devs, v_bins, v_daykeys, _ = data['v']
    user_index = {int(uid): idx for idx, uid in enumerate(data['users'])}
    train_catalog = np.asarray(data['items'], dtype=np.int64)
    all_item_indices = np.arange(len(train_catalog), dtype=np.int64)
    offsets = data['hist_offsets']
    history = data['history_items']
    recommendations = {}
    first_val_row = (
        val.assign(_row=np.arange(len(val)))
           .sort_values(['userId', 'timestamp', 'movieId'])
           .groupby('userId', sort=False)['_row'].first().to_dict()
    )
    model.eval()
    with torch.no_grad():
        implicit = model.implicit_all()
        for j, user_id in enumerate(users, 1):
            ui = user_index[user_id]
            val_row = int(first_val_row[user_id])
            dev = float(v_devs[val_row])
            bin_id = int(v_bins[val_row])
            day_key = int(v_daykeys[val_row])

            already_rated = np.unique(history[offsets[ui]:offsets[ui+1]])
            candidate_idx = np.setdiff1d(all_item_indices, already_rated,
                                         assume_unique=True)
            scores = np.empty(len(candidate_idx), dtype=np.float32)

            for lo in range(0, len(candidate_idx), batch_size):
                hi = min(lo + batch_size, len(candidate_idx))
                n = hi - lo
                U = torch.full((n,), ui, device=device, dtype=torch.long)
                I = torch.as_tensor(candidate_idx[lo:hi], device=device, dtype=torch.long)
                D = torch.full((n,), dev, device=device, dtype=torch.float32)
                B = torch.full((n,), bin_id, device=device, dtype=torch.long)
                K = torch.full((n,), day_key, device=device, dtype=torch.long)
                scores[lo:hi] = model(U, I, D, B, K, implicit).clamp(*rating_scale).cpu().numpy()

            # Deterministic score-descending, movie-ID-ascending tie break.
            candidate_movie_ids = train_catalog[candidate_idx]
            ranking = np.lexsort((candidate_movie_ids, -scores))[:k]
            recommendations[user_id] = candidate_movie_ids[ranking].tolist()
            if verbose and (j % 250 == 0 or j == len(users)):
                print(f'Ranked {j:,}/{len(users):,} users', flush=True)
    return recommendations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/netflix_day_dev')
    parser.add_argument('--checkpoint', default='results/timesvdpp_netflix_best.pt')
    parser.add_argument('--results-json', default='results/timesvdpp_netflix_validation.json')
    parser.add_argument('--smoke-users', type=int, default=0, help='0 = all validation users')
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    set_seed(575)
    start = time.perf_counter()

    data = setup_data(Path(args.data))
    saved = Path(args.results_json)
    if not saved.is_file():
        raise FileNotFoundError(f'Missing training metadata: {saved}')
    config = json.loads(saved.read_text())
    n_factors = int(config['factors'])

    device = torch.device(args.device)
    model = TimeSVDpp(len(data['users']), len(data['items']), data['n_bins'],
                      data['n_day_keys'], data['history_items'], data['hist_offsets'],
                      float(data['t'][-1].mean()), n_factors).to(device)
    try:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    val = data['val']
    selected = sorted(int(uid) for uid in val.userId.unique())
    if args.smoke_users > 0:
        selected = selected[:args.smoke_users]
    if not selected:
        raise ValueError('No validation users selected')

    recommendations = recommend_timesvdpp(model, data, selected, k=20,
                                          batch_size=args.batch_size, device=device)

    heldout = val[val.userId.isin(selected)]
    result = evaluate_ranking(recommendations, heldout, ks=(10, 20))
    result.update({
        'model': 'experimental_TimeSVD++',
        'dataset': str(args.data),
        'checkpoint': str(args.checkpoint),
        'users_ranked': len(selected),
        'catalog_movies': len(data['items']),
        'candidate_protocol': 'train catalog excluding per-user train history',
        'timestamp_protocol': 'first validation day per user, fixed ranking snapshot',
        'smoke_test': args.smoke_users > 0,
        'elapsed_seconds': round(time.perf_counter() - start, 1),
    })
    out = Path('results') / ('timesvdpp_netflix_ranking_smoke.json' if args.smoke_users > 0
                             else 'timesvdpp_netflix_ranking.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print('\n=== VALIDATION RANKING ===')
    for name in ('recall@10', 'ndcg@10', 'recall@20', 'ndcg@20', 'n_users'):
        print(f'{name}: {result[name]:.5f}' if name != 'n_users' else f'{name}: {result[name]}')
    print(f'Saved {out}')
    print('Note: ranking only considers items present in training; unseen validation items cannot be recommended.')


if __name__ == '__main__':
    main()
