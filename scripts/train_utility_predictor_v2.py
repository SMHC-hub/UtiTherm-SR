"""Step 7B2B — train a backward-compatible V2 utility selector using frozen ROI labels.

Keep ALL data in detector TRAIN domain. New sequences are split TRAIN/VAL/TEST;
all V1 sequences are used only for V2 training. V1 and V2 are compared on
new TEST sequences unseen by both utility predictors. These are NOT an
independent detector test (YOLO was trained on these original frames).

Use: python scripts/train_utility_predictor_v2.py --config configs/utility_predictor_v2.yaml --smoke
     python scripts/train_utility_predictor_v2.py --config configs/utility_predictor_v2.yaml
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from train_utility_predictor import load_measurements, ranking_report, weight_by_head
from utitherm.utility.predictor import HEADS, UtilityPredictor, encode_tile, ranking_score

REQUIRED = {'v1_roi_csv', 'fresh_roi_csv', 'train_lr', 'v1_checkpoint', 'output',
            'canonical_checkpoint', 'seed', 'new_val_fraction', 'new_test_fraction',
            'input_width', 'input_height', 'batch_images', 'workers', 'epochs',
            'patience', 'learning_rate', 'weight_decay', 'pos_weight_cap',
            'ranking_loss_weight', 'ranking_logit_scale', 'false_positive_penalty',
            'budget_tiles', 'device'}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for part in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def split_v2(v1_rows: list[dict], fresh_rows: list[dict], seed: int,
             val_fraction: float, test_fraction: float) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Group by sequence; V2's TEST is not used to select a checkpoint."""
    if not (0.0 < val_fraction < 0.4 and 0.0 < test_fraction < 0.4 and val_fraction + test_fraction < .6):
        raise ValueError('Fractions must be positive and leave most fresh sequences for training')
    old_seq = {r['sequence'] for r in v1_rows}
    fresh_seq = {r['sequence'] for r in fresh_rows}
    if old_seq & fresh_seq:
        raise ValueError(f'V1 and fresh sequences overlap: {sorted(old_seq & fresh_seq)[:3]}')
    old_keys = {(r['image'], int(r['tile_index'])) for r in v1_rows}
    fresh_keys = {(r['image'], int(r['tile_index'])) for r in fresh_rows}
    if old_keys & fresh_keys:
        raise ValueError('Repeated images in V1 and new measurements')
    if len(fresh_seq) < 10:
        raise ValueError('Need >=10 fresh sequences to freeze train, val and test groups')
    sequences = sorted(fresh_seq)
    random.Random(seed).shuffle(sequences)
    n_val = max(2, round(len(sequences) * val_fraction))
    n_test = max(2, round(len(sequences) * test_fraction))
    if n_val + n_test >= len(sequences) - 1:
        raise ValueError('Too few training sequences')
    test_seq = set(sequences[:n_test])
    val_seq = set(sequences[n_test:n_test+n_val])
    train_new_seq = set(sequences[n_test+n_val:])
    train = list(v1_rows) + [r for r in fresh_rows if r['sequence'] in train_new_seq]
    val = [r for r in fresh_rows if r['sequence'] in val_seq]
    test = [r for r in fresh_rows if r['sequence'] in test_seq]
    groups = {k: {r['sequence'] for r in rows} for k, rows in [('train', train), ('val', val), ('test', test)]}
    if any(groups[a] & groups[b] for a,b in [('train','val'), ('train','test'), ('val','test')]):
        raise AssertionError('Train/val/test sequence leakage')
    return train, val, test, {k: sorted(v) for k,v in groups.items()}


class ImageTilesDataset(Dataset):
    """Return all 16 interventions for an image so ranking loss is properly grouped."""
    def __init__(self, rows: list[dict], root: Path, size: tuple[int, int]):
        self.root, self.size = root, size
        by_name: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_name[row['image']].append(row)
        self.frames = sorted(by_name)
        self.records = []
        for name in self.frames:
            if not (root / name).is_file():
                raise FileNotFoundError(f'Missing detector-TRAIN image: {root / name}')
            rec = sorted(by_name[name], key=lambda r: int(r['tile_index']))
            if len(rec) != 16 or [int(r['tile_index']) for r in rec] != list(range(16)):
                raise ValueError(f'Incomplete 16-tile records: {name}')
            self.records.append(rec)

    @lru_cache(maxsize=96)
    def _gray(self, name: str) -> np.ndarray:
        img = cv2.imread(str(self.root / name), cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape != (256, 320):
            raise ValueError(f'Expected 320x256 grayscale LR: {name}')
        return img

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i: int):
        image = self._gray(self.frames[i])
        rec = self.records[i]
        encoded = [encode_tile(image, row['rect'], input_size=self.size) for row in rec]
        x = torch.stack([item[0] for item in encoded])
        coord = torch.stack([item[1] for item in encoded])
        labels = torch.from_numpy(np.stack([r['target'] for r in rec]).astype(np.float32))
        utility = torch.tensor([float(r['score']) for r in rec], dtype=torch.float32)
        return x, coord, labels, utility

    def flat_rows(self) -> list[dict]:
        return [r for group in self.records for r in group]


def torch_rank_score(logits: torch.Tensor, penalty: float) -> torch.Tensor:
    p = logits.sigmoid()
    return p[..., 0] - p[..., 2] + penalty * (p[..., 1] - p[..., 3])


def pairwise_ranking_loss(logits: torch.Tensor, actual: torch.Tensor, penalty: float,
                          scale: float) -> torch.Tensor:
    """Reward ranking good regions ahead of harmful ones, ignore tied scores.

    logits [B, 16, 4], actual marginal ROI utility [B, 16].
    """
    if logits.ndim != 3 or logits.shape[-2:] != (16, 4) or actual.shape != logits.shape[:2]:
        raise ValueError('Expected [batch,16,4] and [batch,16]')
    score = torch_rank_score(logits, penalty)
    dy = actual.unsqueeze(2) - actual.unsqueeze(1)
    ds = score.unsqueeze(2) - score.unsqueeze(1)
    upper = torch.triu(torch.ones((16,16), dtype=torch.bool, device=actual.device), diagonal=1)
    valid = (dy.abs() > 1e-6) & upper.unsqueeze(0)
    if not valid.any():
        return logits.sum() * 0.0
    # Emphasize larger counterfactual differences while capping outlier influence.
    weight = dy.abs().clamp(max=2.0)
    losses = F.softplus(-scale * dy.sign() * ds) * weight
    return (losses * valid).sum() / (weight * valid).sum().clamp_min(1e-6)


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
              criterion: nn.Module, penalty: float, rank_weight: float, rank_scale: float,
              optimizer: torch.optim.Optimizer | None = None) -> tuple[dict, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    stats = np.zeros(3, dtype=np.float64)
    total = 0
    predictions = []
    labels_list = []
    with torch.enable_grad() if training else torch.inference_mode():
        for x, coords, labels, utility in loader:
            x, coords, labels, utility = [v.to(device, non_blocking=True) for v in (x, coords, labels, utility)]
            B, T, C, H, W = x.shape
            logits = model(x.reshape(B*T, C, H, W), coords.reshape(B*T, 4)).reshape(B, T, 4)
            bce = criterion(logits, labels)
            rank = pairwise_ranking_loss(logits, utility, penalty, rank_scale)
            loss = bce + rank_weight * rank
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            stats += np.array([float(loss.detach()), float(bce.detach()), float(rank.detach())]) * B
            total += B
            if not training:
                predictions.append(logits.sigmoid().cpu().numpy().reshape(-1, 4))
                labels_list.append(labels.cpu().numpy().reshape(-1, 4))
    if not total:
        raise ValueError('Empty loader')
    return dict(zip(('total','bce','ranking'), (stats/total).tolist())), (
        np.concatenate(predictions) if predictions else np.empty((0, 4))), (
        np.concatenate(labels_list) if labels_list else np.empty((0,4)))


def head_metrics(y_true: np.ndarray, p: np.ndarray) -> dict:
    result = {}
    for i, head in enumerate(HEADS):
        npos = int(y_true[:, i].sum())
        result[head] = {
            'positive_count': npos,
            'positive_rate': round(float(y_true[:,i].mean()), 6),
            'average_precision': round(float(average_precision_score(y_true[:,i], p[:,i])), 6)
                                 if 0 < npos < len(y_true) else None,
        }
    return result


def topk_per_image(rows: list[dict], probabilities: np.ndarray, budget: int,
                   penalty: float) -> dict[str, dict]:
    score = ranking_score(probabilities, fp_penalty=penalty)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['image']].append(i)
    selections = {}
    for name, indices in groups.items():
        if len(indices) != 16:
            raise ValueError('Need 16 records per image')
        selected = sorted(indices, key=lambda i: (-float(score[i]), int(rows[i]['tile_index'])))[:budget]
        selections[name] = {
            'sequence': rows[selected[0]]['sequence'],
            'utility': float(sum(rows[i]['score'] for i in selected)),
            'small_gains': int(sum(rows[i]['target'][0] for i in selected)),
            'small_losses': int(sum(rows[i]['target'][2] for i in selected)),
            'fp_reductions': int(sum(rows[i]['target'][1] for i in selected)),
            'fp_increases': int(sum(rows[i]['target'][3] for i in selected)),
            'tiles': sorted(int(rows[i]['tile_index']) for i in selected)
        }
    return selections


def sequence_bootstrap_delta(v1: dict, v2: dict, seed: int, draws: int = 1500) -> dict:
    if set(v1) != set(v2) or not v1:
        raise ValueError('V1 and V2 must score exactly the same frozen images')
    groups = defaultdict(list)
    for name in sorted(v1):
        if v1[name]['sequence'] != v2[name]['sequence']:
            raise ValueError('Sequence mismatch')
        groups[v1[name]['sequence']].append(v2[name]['utility'] - v1[name]['utility'])
    if len(groups) < 2:
        raise ValueError('At least 2 sequences required for bootstrap')
    seq_means = np.array([np.mean(groups[k]) for k in sorted(groups)], dtype=np.float64)
    point = float(np.mean(seq_means))
    rng = np.random.default_rng(seed)
    samples = rng.choice(seq_means, size=(draws, len(seq_means)), replace=True).mean(axis=1)
    return {'delta_v2_minus_v1_per_sequence_mean': round(point, 6),
            'sequence_bootstrap_exploratory_95ci': [round(float(v),6) for v in np.quantile(samples,[.025,.975])],
            'num_sequences': len(groups),
            'note': 'Sequence-cluster bootstrap on single-tile marginal ranking proxies, NOT true multi-ROI detection gain.'}


def fresh_data_sanity(old_rows: list[dict], new_rows: list[dict]) -> None:
    # Do not train on labels that may have changed their outcome convention.
    for row in old_rows + new_rows:
        if set(row.keys()) < {'image', 'sequence', 'rect', 'target', 'score'}:
            raise ValueError('ROI label row lacks required fields')


def load_v1(path: Path, device: torch.device, size: tuple[int,int]) -> nn.Module:
    if not path.is_file():
        raise FileNotFoundError(f'Missing frozen V1 model: {path}')
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if tuple(ckpt.get('heads', [])) != tuple(HEADS) or tuple(ckpt.get('input_size', [])) != size:
        raise ValueError('V1 checkpoint heads/input size incompatible with current inference model')
    model = UtilityPredictor().to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=Path('configs/utility_predictor_v2.yaml'))
    ap.add_argument('--smoke', action='store_true', help='Small train/val/test group test, one epoch; do not install checkpoint')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    if not args.config.is_file():
        ap.error(f'Config does not exist: {args.config}')
    conf = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if not isinstance(conf, dict) or REQUIRED - set(conf):
        ap.error(f'Missing configuration keys: {sorted(REQUIRED - set(conf if isinstance(conf,dict) else {}))}')
    out = args.out or Path(str(conf['output']) + ('_smoke' if args.smoke else ''))
    if out.exists():
        ap.error(f'Experiment already exists: {out}. Set a new --out; do not overwrite')
    root = Path(conf['train_lr'])
    if root.name != 'train' or root.parent.name != 'images' or 'yolo_flir_lr_x2' not in root.parts:
        ap.error('train_lr MUST be the YOLO FLIR LR detector-TRAIN folder, never VAL/test')
    if not root.is_dir():
        ap.error(f'Missing training LR folder: {root}')
    penalty = float(conf['false_positive_penalty'])
    if abs(penalty - .25) > 1e-9:
        ap.error('This experiment requires exact Step 5A / 7B2A FP penalty of 0.25')
    old_path, fresh_path = Path(conf['v1_roi_csv']), Path(conf['fresh_roi_csv'])
    old_rows = load_measurements(old_path, penalty)
    new_rows = load_measurements(fresh_path, penalty)
    fresh_data_sanity(old_rows, new_rows)
    if args.smoke:
        # Sequence-level subset; three-way split remains valid and V1 is frozen.
        old_ids = set(sorted({r['sequence'] for r in old_rows})[:8])
        new_ids = set(sorted({r['sequence'] for r in new_rows})[:12])
        old_rows = [r for r in old_rows if r['sequence'] in old_ids]
        new_rows = [r for r in new_rows if r['sequence'] in new_ids]
    seed = int(conf['seed'])
    train, val, test, split = split_v2(old_rows, new_rows, seed,
                                      float(conf['new_val_fraction']), float(conf['new_test_fraction']))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    request = str(conf['device']).lower()
    if request not in ('cpu', 'auto', 'cuda', '0'):
        ap.error('device must be cpu, auto, cuda or 0')
    use_gpu = request in ('cuda', 'auto', '0') and torch.cuda.is_available()
    if request in ('cuda', '0') and not use_gpu:
        ap.error('CUDA requested but not available')
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    if use_gpu:
        torch.backends.cudnn.benchmark = False
    size = (int(conf['input_width']), int(conf['input_height']))
    if min(size) < 16 or size != (80, 64):
        ap.error('V2 must retain V1 80×64 encoding for checkpoint compatibility')
    datasets = {k: ImageTilesDataset(rows, root, size)
                for k, rows in [('train',train), ('val',val), ('test',test)]}
    if min(len(ds) for ds in datasets.values()) < 2:
        ap.error('At least two images in each split are required')
    batch = int(conf['batch_images'])
    if not 1 <= batch <= 8:
        ap.error('batch_images must be between 1 and 8 to respect the Docker/GPU memory budget')
    workers = int(conf['workers'])
    loaders = {k: DataLoader(ds, batch_size=batch, shuffle=(k=='train'),
                             num_workers=workers, pin_memory=use_gpu)
               for k,ds in datasets.items()}
    v1 = load_v1(Path(conf['v1_checkpoint']), device, size)
    model = UtilityPredictor().to(device)
    weights = weight_by_head(train, float(conf['pos_weight_cap'])).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=float(conf['learning_rate']), weight_decay=float(conf['weight_decay']))
    rank_weight = float(conf['ranking_loss_weight'])
    rank_scale = float(conf['ranking_logit_scale'])
    if rank_weight < 0 or rank_scale <= 0 or not 1 <= int(conf['budget_tiles']) <= 16:
        ap.error('Invalid ranking weight/scale/budget')
    epochs = 1 if args.smoke else int(conf['epochs'])
    patience = int(conf['patience'])
    if epochs < 1 or patience < 1:
        ap.error('epochs/patience must be positive')
    print(f'V2 {device} | images train/val/test: {len(datasets["train"])}/{len(datasets["val"])}/{len(datasets["test"])} '
          f'| sequences: {len(split["train"])}/{len(split["val"])}/{len(split["test"])} '
          f'| parameters: {sum(p.numel() for p in model.parameters()):,}', flush=True)
    out.mkdir(parents=True)
    hashes = {str(p): sha256(p) for p in (old_path, fresh_path, Path(conf['v1_checkpoint']))}
    (out/'split_manifest.json').write_text(json.dumps({'seed': seed, 'sequences': split,
        'image_counts': {k: len(ds) for k,ds in datasets.items()}, 'input_hashes': hashes,
        'test_selection_policy': 'new sequence groups; never used for V2 checkpoint selection'}, indent=2)+'\n',encoding='utf8')
    history, best, stale = [], float('inf'), 0
    for epoch in range(1, epochs + 1):
        tr, _, _ = run_epoch(model, loaders['train'], device, criterion, penalty, rank_weight, rank_scale, opt)
        va, _, _ = run_epoch(model, loaders['val'], device, criterion, penalty, rank_weight, rank_scale)
        history.append({'epoch': epoch, 'train_total': tr['total'], 'val_total': va['total'],
                        'train_bce': tr['bce'], 'val_bce': va['bce'],
                        'train_ranking': tr['ranking'], 'val_ranking': va['ranking']})
        if va['total'] < best - 1e-6:
            best, stale = va['total'], 0
            torch.save({'model':model.state_dict(), 'heads':list(HEADS), 'input_size':list(size),
                        'seed':seed, 'epoch':epoch, 'source_csv_sha256':sha256(fresh_path),
                        'comment':'Step 7B2B V2 utility; development, NOT independent detector evidence'}, out/'best.pt')
        else:
            stale += 1
        print(f'epoch {epoch:02}/{epochs}: train total={tr["total"]:.4f} '
              f'val total={va["total"]:.4f} (BCE={va["bce"]:.4f}, rank={va["ranking"]:.4f}) best={best:.4f}', flush=True)
        if stale >= patience:
            print('Early stopping (new-sequence validation loss did not improve)', flush=True)
            break
    best_state = torch.load(out/'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(best_state['model'], strict=True)
    # TEST untouched until AFTER last epoch, early stopping, and checkpoint loading.
    test_criterion = criterion
    test_v2, p2, labels = run_epoch(model, loaders['test'], device, test_criterion, penalty, rank_weight, rank_scale)
    test_v1, p1, _ = run_epoch(v1, loaders['test'], device, test_criterion, penalty, rank_weight, rank_scale)
    ordered_rows = datasets['test'].flat_rows()
    budget = int(conf['budget_tiles'])
    rank1 = ranking_report(ordered_rows, p1, budget, seed, penalty)
    rank2 = ranking_report(ordered_rows, p2, budget, seed, penalty)
    sel1, sel2 = (topk_per_image(ordered_rows, p, budget, penalty) for p in (p1,p2))
    comparison = sequence_bootstrap_delta(sel1, sel2, seed)
    with (out/'history.csv').open('w',encoding='utf8',newline='') as fh:
        cols = list(history[0])
        writer = csv.DictWriter(fh, fieldnames=cols); writer.writeheader(); writer.writerows(history)
    with (out/'test_rankings.csv').open('w',encoding='utf8',newline='') as fh:
        cols = ['image','sequence','v1_tiles','v2_tiles','v1_marginal_utility','v2_marginal_utility',
                'v1_small_gains','v2_small_gains','v1_small_losses','v2_small_losses','v1_fp_increases','v2_fp_increases']
        writer = csv.DictWriter(fh,fieldnames=cols); writer.writeheader()
        for name in sorted(sel1):
            a, b = sel1[name], sel2[name]
            writer.writerow({'image':name, 'sequence':a['sequence'], 'v1_tiles':','.join(map(str,a['tiles'])),
                'v2_tiles':','.join(map(str,b['tiles'])), 'v1_marginal_utility':a['utility'],
                'v2_marginal_utility':b['utility'], 'v1_small_gains':a['small_gains'],
                'v2_small_gains':b['small_gains'], 'v1_small_losses':a['small_losses'],
                'v2_small_losses':b['small_losses'], 'v1_fp_increases':a['fp_increases'],
                'v2_fp_increases':b['fp_increases']})
    result = {'task':'Step 7B2B — ROI Utility Predictor V2 (train-domain only)',
              'utc':datetime.now(timezone.utc).isoformat(), 'device':str(device),
              'v1_checkpoint':str(conf['v1_checkpoint']), 'v2_checkpoint':str(out/'best.pt'),
              'best_epoch':best_state['epoch'], 'best_validation_combined_loss':best,
              'sequences':{k:len(s) for k,s in split.items()},
              'images':{k:len(ds) for k,ds in datasets.items()},
              'test':{'v1':{'loss':test_v1,'head_ap':head_metrics(labels,p1),'ranking_proxy':rank1},
                      'v2':{'loss':test_v2,'head_ap':head_metrics(labels,p2),'ranking_proxy':rank2},
                      'paired_rank_delta':comparison},
              'limitations':['Predictor V2 model selection used fresh-sequence VAL; comparison uses separate fresh-sequence TEST.',
                  'Both V1 and V2 TEST images were in YOLO detector TRAIN data, not independent detection tests.',
                  'Measured utility is sum of SINGLE ROI counterfactuals, not actual combined-ROI TP/FP.',
                  'V2 still predicts utility for Bicubic interventions, not FSRCNN.',
                  'Model checkpoint is backwards-compatible with original 4-head inference, but speed is not improved by this training change.',
                  'Do not tune model based on TEST repeatedly; use independent FLIR validation for later end-to-end development only.',
                  'Detector out-of-fold labels and genuinely untouched external final test remain required.']}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    if not args.smoke:
        canonical=Path(conf['canonical_checkpoint'])
        if canonical.exists():
            print(f'Canonical V2 exists, NOT overwriting: {canonical}',flush=True)
        else:
            canonical.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(out/'best.pt',canonical)
            print(f'Canonical V2 checkpoint: {canonical}',flush=True)
    print(f'DONE — {out/"summary.json"} | test top-{budget} marginal utility V1='
          f'{rank1["mean_marginal_utility_per_image"]["predicted"]:.3f} '
          f'V2={rank2["mean_marginal_utility_per_image"]["predicted"]:.3f} '
          '(PROXY, not observed combined-region detector improvement)',flush=True)


if __name__ == '__main__':
    main()
