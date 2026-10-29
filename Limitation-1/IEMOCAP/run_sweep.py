"""
Run train_aux.py for every (config, seed), spreading runs over the available GPUs.
Finished runs are skipped, so re-running after a timeout resumes where it stopped.

    python run_sweep.py --data_dir DATA --output_dir OUT --configs CTRL,VAD,SOFT,BOTH --seeds 1-10 --gpus 0,1
"""
import argparse
import os
import subprocess
import sys
import threading
import queue

CONFIGS = {
    # final comparison
    'CTRL': [],
    'VAD': ['--lambda_vad', '0.3'],
    'SOFT': ['--soft_beta', '0.5'],
    'BOTH': ['--lambda_vad', '1.0', '--soft_beta', '0.5'],
    # stage-1 grid (pick the best by mean VALID wF1, never by test)
    'VAD_l0.1': ['--lambda_vad', '0.1'],
    'VAD_l1.0': ['--lambda_vad', '1.0'],
    'VAD_A_l0.3': ['--lambda_vad', '0.3', '--vad_dims', 'a'],   # arousal only
    'SOFT_b0.3': ['--soft_beta', '0.3'],
    'SOFT_b1.0': ['--soft_beta', '1.0'],
    # WavLM audio instead of IS10 (W_* configs get --audio_feat from the command line)
    'W_CTRL': [],
    'W_VAD': ['--lambda_vad', '1.0'],
    'W_SOFT': ['--soft_beta', '0.5'],
    'W_BOTH': ['--lambda_vad', '1.0', '--soft_beta', '0.5'],
}


def parse_seeds(s):
    if '-' in s:
        a, b = s.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(',')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--configs', default='CTRL,VAD,SOFT,BOTH')
    p.add_argument('--seeds', default='1-10')
    p.add_argument('--gpus', default='0', help="comma-separated GPU ids, e.g. '0,1' on Kaggle T4 x2")
    p.add_argument('--extra', default='', help='extra args passed to every run, e.g. "--save_ckpt"')
    p.add_argument('--audio_feat', default=None, help='WavLM pkl from extract_wavlm.py, used by the W_* configs')
    args = p.parse_args()
    if any(c.startswith('W_') for c in args.configs.split(',')) and not args.audio_feat:
        p.error('W_* configs need --audio_feat')

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
            cmd = [sys.executable, os.path.join(here, 'train_aux.py'), '--data_dir', args.data_dir,
                   '--output_dir', args.output_dir, '--tag', cfg, '--seed', str(seed)] + CONFIGS[cfg] + args.extra.split()
            if cfg.startswith('W_'):
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
