"""
Run train_merged.py for every (config, seed), spreading runs over the available GPUs.
Finished runs are skipped, so re-running after a timeout resumes where it stopped.

    python run_sweep.py --data_dir DATA --output_dir OUT --configs CTRL,W_VAD,CACG,CACG_W_VAD --seeds 1-20 \
                        --gpus 0,1 --audio_feat iemocap_wavlm.pkl
"""
import argparse
import os
import subprocess
import sys
import threading
import queue

CACG = ['--gate', 'cacg', '--gate_loss', 'none']   # Lakshay's IEMOCAP setting: dynamic tau, no gate loss
VAD = ['--lambda_vad', '1.0']                     # Limitation 1 choice from Round 2

CONFIGS = {
    # final comparison
    'CTRL': [],                      # original ECERC, IS10 audio
    'W_VAD': VAD,                    # Limitation 1 alone: WavLM audio + VAD supervision
    'CACG': CACG,                    # Limitation 3 alone
    'CACG_W_VAD': CACG + VAD,        # merged model
    # optional ablations
    'W_CTRL': [],                    # WavLM audio only
    'CACG_W': CACG,                  # CACG + WavLM audio, no VAD
    'CACG_UTIL_W_VAD': ['--gate', 'cacg', '--gate_loss', 'utility', '--aux_heads',
                        '--lambda_gate', '1.0', '--lambda_aux', '0.5'] + VAD,   # Lakshay's MELD setting
}
WAVLM_CONFIGS = {'W_VAD', 'CACG_W_VAD', 'W_CTRL', 'CACG_W', 'CACG_UTIL_W_VAD'}


def parse_seeds(s):
    if '-' in s:
        a, b = s.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(',')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--configs', default='CTRL,W_VAD,CACG,CACG_W_VAD')
    p.add_argument('--seeds', default='1-20')
    p.add_argument('--gpus', default='0', help="comma-separated GPU ids, e.g. '0,1' on Kaggle T4 x2")
    p.add_argument('--extra', default='', help='extra args passed to every run, e.g. "--save_ckpt"')
    p.add_argument('--audio_feat', default=None, help='WavLM pkl from extract_wavlm.py, used by the WavLM configs')
    args = p.parse_args()
    if any(c in WAVLM_CONFIGS for c in args.configs.split(',')) and not args.audio_feat:
        p.error('WavLM configs ({}) need --audio_feat'.format(','.join(sorted(WAVLM_CONFIGS))))

    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    jobs = queue.Queue()
    for seed in parse_seeds(args.seeds):          # seed-major order: all configs of a seed finish together
        for cfg in args.configs.split(','):
            jobs.put((cfg, seed))

    here = os.path.dirname(os.path.abspath(__file__))

    def worker(gpu):
        while True:
            try:
                cfg, seed = jobs.get_nowait()
            except queue.Empty:
                return
            log = os.path.join(args.output_dir, 'logs', '{}_s{}.log'.format(cfg, seed))
            cmd = [sys.executable, os.path.join(here, 'train_merged.py'), '--data_dir', args.data_dir,
                   '--output_dir', args.output_dir, '--tag', cfg, '--seed', str(seed)] + CONFIGS[cfg] + args.extra.split()
            if cfg in WAVLM_CONFIGS:
                cmd += ['--audio_feat', args.audio_feat]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            with open(log, 'w') as f:
                rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=here, env=env)
            last = open(log).read().strip().splitlines()[-1:] or ['']
            print('[gpu {}] {}_s{} rc={} | {}'.format(gpu, cfg, seed, rc, last[0]), flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus.split(',')]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == '__main__':
    main()
