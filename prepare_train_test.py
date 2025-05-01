import os
import argparse
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import resample

def load_mat(matfile):
    m = scipy.io.loadmat(matfile)
    feeg = m['specDat'][0]['fEEG'][0]
    time = feeg[0]      
    eeg  = feeg[1].astype(float)
    sr   = 1.0 / np.diff(time[:2])[0]
    events = m['szEvents']   # [idx, start_s, end_s, dur_s]
    return eeg, sr, events

def to_idx(t, sr):
    return int(round(t * sr))

def sliding_windows(arr, win_len):
    n = len(arr)
    count = n // win_len
    return [ arr[i*win_len:(i+1)*win_len] for i in range(count) ]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--matfile',    required=True)
    p.add_argument('--window-sec', type=float, default=10.0)
    p.add_argument('--preictal-sec', type=float, default=5.0)
    p.add_argument('--resample-hz', type=float, default=None)
    p.add_argument('--train-ratio', type=float, default=0.8)
    args = p.parse_args()

    eeg, sr, events = load_mat(args.matfile)
    # optional resampling
    if args.resample_hz:
        new_len = int(len(eeg) * args.resample_hz / sr)
        eeg = resample(eeg, new_len)
        sr = args.resample_hz

    W = int(round(args.window_sec * sr))
    P = int(round(args.preictal_sec * sr))

    # build interictal intervals
    seiz = sorted((to_idx(s,sr), to_idx(e,sr)) for _, s,e,_ in events)
    inter = []
    last = 0
    for s,e in seiz:
        if s - last >= W:
            inter.append((last, s))
        last = e
    if len(eeg) - last >= W:
        inter.append((last, len(eeg)))

    # windows
    inter_wins = []
    for a,b in inter:
        inter_wins += sliding_windows(eeg[a:b], W)

    # preictal → ictal windows (windows ending at seizure start)
    seiz_wins = []
    for s,e in seiz:
        if s >= W:
            segment = eeg[s-W:s]  # last W samples before seizure
            # ensure its second half overlaps the seizure
            seiz_wins.append(segment)

    # balance: down-sample interictal to match seiz count
    N = min(len(inter_wins), len(seiz_wins))
    np.random.shuffle(inter_wins)
    inter_wins = inter_wins[:N]
    seiz_wins = seiz_wins[:N]

    # label-free merge and split
    all_wins = np.array(inter_wins + seiz_wins)
    np.random.shuffle(all_wins)
    cut = int(args.train_ratio * len(all_wins))
    train, test = all_wins[:cut], all_wins[cut:]

    # save
    os.makedirs('data', exist_ok=True)
    pd.DataFrame(train).to_csv('data/train.csv', index=False, header=False)
    pd.DataFrame(test).to_csv('data/test.csv', index=False, header=False)
    print(f"train: {len(train)} windows, test: {len(test)} windows")

if __name__=='__main__':
    main()

