"""Writes every architecture diagram of the final report as an SVG next to this script.
Inline attributes only and arrowheads as polygons, so the files also import into PowerPoint.

    python make_svgs.py
"""
import math
import os
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
W = 760
FONT = 'Arial, Helvetica, sans-serif'
INK, SUB, LINE = '#1F2933', '#3E4A56', '#3D4852'
N = ('#F3F5F7', '#66717E')
PLAIN = ('#FFFFFF', '#66717E')
L1 = ('#DFF3EF', '#0F766E')
L2 = ('#FCEBD7', '#B45309')
L3 = ('#E9E8FB', '#4338CA')
L3W = ('#FFFFFF', '#4338CA')


class Svg:
    def __init__(self):
        self.back, self.lines, self.front = [], [], []

    def rect(self, x, y, w, h, c, dash=False, layer=None, sw=1.5):
        d = ' stroke-dasharray="7 5"' if dash else ''
        (self.front if layer is None else layer).append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" ry="8" fill="{c[0]}" stroke="{c[1]}" '
            f'stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=12, bold=False, fill=INK, anchor='middle'):
        b = ' font-weight="bold"' if bold else ''
        self.front.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{fill}" '
                          f'text-anchor="{anchor}"{b}>{escape(s)}</text>')

    def box(self, x, y, w, h, c, lines, dash=False):
        """First line is the bold name, the rest are body text; centred in the box."""
        self.rect(x, y, w, h, c, dash)
        lh = 17
        y0 = y + (h - lh * len(lines)) / 2 + 13
        for i, s in enumerate(lines):
            self.text(x + w / 2, y0 + i * lh, s, 13.5 if i == 0 else 12, i == 0, INK if i == 0 else SUB)

    def arrow(self, pts, dash=False, color=LINE):
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        a = math.atan2(y2 - y1, x2 - x1)
        bx, by = x2 - 8 * math.cos(a), y2 - 8 * math.sin(a)
        d = 'M ' + ' L '.join(f'{x} {y}' for x, y in pts[:-1]) + f' L {bx:.1f} {by:.1f}'
        ds = ' stroke-dasharray="6 4"' if dash else ''
        self.lines.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.5"{ds}/>')
        px, py = -math.sin(a) * 4.5, math.cos(a) * 4.5
        self.lines.append(f'<polygon points="{x2},{y2} {bx + px:.1f},{by + py:.1f} {bx - px:.1f},{by - py:.1f}" '
                          f'fill="{color}"/>')

    def key(self, items, y=46):
        x = 24
        for c, label in items:
            self.front.append(f'<rect x="{x}" y="{y}" width="14" height="14" rx="3" ry="3" fill="{c[0]}" '
                              f'stroke="{c[1]}" stroke-width="1.5"/>')
            self.text(x + 20, y + 11, label, 12, fill=SUB, anchor='start')
            x += 34 + len(label) * 6.6

    def write(self, name, title, height):
        desc = escape(title)
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" width="{W}" height="{height}" '
               f'role="img" aria-label="{desc}">', f'<title>{desc}</title>',
               f'<rect x="0" y="0" width="{W}" height="{height}" fill="#FFFFFF"/>']
        out += self.back + self.lines + self.front + ['</svg>']
        with open(os.path.join(HERE, name), 'w', encoding='utf-8') as f:
            f.write('\n'.join(out) + '\n')
        print('wrote', name)


def backbone(s, audio, audio_c):
    """Inputs to Feature Gating; shared by every diagram. Ends at y = 530."""
    s.box(24, 72, 116, 64, N, ['Text t', 'Emotion RoBERTa'])
    s.box(151, 72, 116, 64, audio_c, ['Audio a'] + audio)
    s.box(278, 72, 116, 64, N, ['Visual v', '342-d'])
    s.box(414, 72, 146, 64, N, ['Semantic s', 'RoBERTa, 1024-d'])
    s.box(586, 72, 150, 64, N, ['Speaker IDs', 'Who said each turn'])
    s.box(24, 160, 370, 56, N, ['Evidence Gating', 'Cross-modal sigmoid gates, 128-d each, 384-d output'])
    s.box(414, 160, 146, 56, N, ['Projection', '1024-d to 128-d'])
    s.box(24, 240, 226, 56, N, ['Emotion Encoder', 'Key and value for f1, f2'])
    s.box(414, 240, 146, 56, N, ['Event Encoder', 'Key, value for f3, f4'])
    s.rect(24, 330, 712, 120, ('#FAFBFC', N[1]), layer=s.back)
    s.text(380, 352, 'Evidence-Cause Interaction: 4 cross-attentions', 13.5, True)
    for x, a, b, c in [(40, 'f1 Self-contagion', "Same speaker's", 'past turns'),
                       (213, 'f2 Cross-emotion', "Other speakers'", 'past turns'),
                       (386, 'f3 Self-event', "The utterance's", 'own event'),
                       (559, 'f4 Cross-event', "Other speakers'", 'past events')]:
        s.box(x, 362, 161, 72, PLAIN, [a, b, c])
    s.box(24, 474, 712, 56, N, ['Feature Gating', 'Element-wise sigmoid gate on each cause, 384-d each'])
    for pts in [[(82, 136), (82, 160)], [(209, 136), (209, 160)], [(336, 136), (336, 160)], [(487, 136), (487, 160)],
                [(487, 216), (487, 240)], [(137, 216), (137, 240)], [(137, 296), (137, 330)], [(336, 216), (336, 330)],
                [(487, 296), (487, 330)], [(661, 136), (661, 330)], [(380, 450), (380, 474)]]:
        s.arrow(pts)
    s.text(344, 316, 'Evidence as query', 11.5, fill=SUB, anchor='start')
    s.text(653, 316, 'Speaker masks', 11.5, fill=SUB, anchor='end')


def cause_stage(s, cacg):
    """Returns the y where the stage ends."""
    s.arrow([(380, 530), (380, 554)])
    if not cacg:
        s.box(24, 554, 712, 56, N, ['Concatenate the four causes', '4 × 384 = 1536-d fused feature R'])
        return 610
    s.rect(24, 554, 712, 188, L3, layer=s.back)
    s.text(380, 576, 'CACG cause gate', 13.5, True)
    s.box(40, 586, 161, 64, L3W, ['Descriptor h', '12-d evidence summary'])
    s.box(213, 586, 161, 64, L3W, ['Temperature τ', 'Between 0.25 and 4'])
    s.box(386, 586, 161, 64, L3W, ['Scorer z', 'One logit per cause'])
    s.box(559, 586, 161, 64, L3W, ['Availability mask', 'No history: weight 0'])
    s.box(40, 670, 680, 56, L3W, ['Cause weights g = softmax(z / τ)',
                                  'Fused feature R = [4g1 · f1 ; … ; 4g4 · f4], 1536-d'])
    s.arrow([(24, 188), (12, 188), (12, 618), (40, 618)], dash=True)
    s.arrow([(201, 618), (213, 618)])
    for x in (293.5, 466.5, 639.5):
        s.arrow([(x, 650), (x, 670)])
    return 742


def stack(name, title, audio, audio_c, cacg, classifier, train=None, posts=(), out=(), keys=()):
    """One dataset route. train = (name, lines) for a training-only side box; a post with side=... gets a source box."""
    s = Svg()
    s.text(24, 32, title, 15, True, anchor='start')
    s.key(keys)
    backbone(s, audio, audio_c)
    y = cause_stage(s, cacg)
    two_col = train is not None or any(p.get('side') for p in posts)
    x, w = 24, 440 if two_col else 712
    mid = x + w / 2
    cy = y + 24
    s.arrow([(mid, y), (mid, cy)])
    s.box(x, cy, w, 56, N, ['Classifier', classifier])
    if train:
        s.box(488, cy, 248, 72, L3 if train[2] == 'L3' else L1, [train[0]] + train[1], dash=True)
        s.arrow([(612, y), (612, cy)], dash=True)
    y = cy + 56
    for p in posts:
        cy = y + 24
        s.arrow([(mid, y), (mid, cy)])
        s.box(x, cy, w, 56, p['c'], [p['name'], p['body']])
        if p.get('side'):
            s.box(488, cy - 8, 248, 72, p['c'], p['side'])
            s.arrow([(488, cy + 28), (x + w, cy + 28)])
        y = cy + 56
    cy = y + 24
    s.arrow([(mid, y), (mid, cy)])
    h = 26 + 17 * len(out)
    s.box(24, cy, 712, h, PLAIN, ['Output for each utterance'] + list(out))
    s.write(name, title, int(cy + h + 24))


def unified():
    s = Svg()
    title = 'Approach B: all three limitations in one rewritten codebase'
    s.text(24, 32, title, 15, True, anchor='start')
    s.key([(N, 'ECERC paper'), (L1, 'L1 Divyansh Dubey'), (L2, 'L2 Keshav Mishra'), (L3, 'L3 Lakshay Gupta')])
    backbone(s, ['IEMOCAP: WavLM', 'MELD: 300-d'], L1)
    y = cause_stage(s, True)
    s.arrow([(380, y), (380, y + 24)])
    s.box(24, 766, 712, 56, N, ['Classifier', 'Linear layer: 6 emotions on IEMOCAP, 7 on MELD'])
    s.text(736, 846, 'After training: w and T are set on validation only', 11.5, fill=SUB, anchor='end')
    s.box(24, 858, 226, 64, L3, ['IEMOCAP: 5 seeds', '÷ T per seed, softmax, then mean'])
    s.box(274, 858, 142, 64, L3, ['MELD: 3 seeds', 'Softmax, then mean'])
    s.box(434, 858, 142, 64, L2, ['LLM fusion', '0.8 model + 0.2 LLM'])
    s.box(594, 858, 142, 64, L3, ['Calibrate', '÷ T = 1.64'])
    s.box(434, 946, 142, 64, L2, ['Qwen2.5-32B', 'Zero-shot, cached'])
    s.box(24, 1034, 712, 72, PLAIN, ['Output for each utterance',
                                     'Emotion, calibrated confidence, low-confidence flag and cause weights',
                                     'IEMOCAP: wF1 72.77, mF1 71.52, ECE 2.50. MELD: wF1 67.10, mF1 53.82, ECE 2.99.'])
    for pts in [[(137, 822), (137, 858)], [(345, 822), (345, 858)], [(416, 890), (434, 890)], [(576, 890), (594, 890)],
                [(505, 946), (505, 922)], [(137, 922), (137, 1034)], [(665, 922), (665, 1034)]]:
        s.arrow(pts)
    s.write('approach_b_unified.svg', title, 1130)


def overview():
    s = Svg()
    title = 'How the project fits together: three solutions, two merges'
    s.text(24, 32, title, 15, True, anchor='start')
    s.key([(N, 'Shared base'), (L1, 'L1 Divyansh Dubey'), (L2, 'L2 Keshav Mishra'), (L3, 'L3 Lakshay Gupta')])
    s.box(24, 72, 712, 56, N, ['ECERC release', "The authors' code, precomputed features and trained checkpoints"])
    s.rect(24, 152, 712, 148, ('#FAFBFC', N[1]), layer=s.back)
    s.text(380, 174, 'Individual solutions', 13.5, True)
    s.box(40, 186, 216, 98, L1, ['L1: category confusion', 'IEMOCAP', "WavLM audio + VAD head", "Authors' code, 20 seeds"])
    s.box(272, 186, 216, 98, L3, ['L3: cause gating', 'Both datasets', 'CACG + temperature scaling', 'Rewritten code'])
    s.box(504, 186, 216, 98, L2, ['L2: class imbalance', 'MELD', 'Zero-shot LLM fusion', "Authors' checkpoints"])
    s.arrow([(380, 128), (380, 152)])
    s.arrow([(197, 300), (197, 324)])
    s.arrow([(563, 300), (563, 324)])
    for x, name, a, b, c, d, e in [
        (24, 'Approach A: merged separately', 'IEMOCAP: L1 + L3', "Authors' code, 20 seeds", 'Ensemble wF1 72.43, mF1 71.14',
         'MELD: L2 + L3', 'Rewritten code copy, 3 seeds'),
        (390, 'Approach B: one codebase', 'IEMOCAP: L1 + L3', 'Rewritten code, 5 seeds', 'Ensemble wF1 72.77, mF1 71.52',
         'MELD: L2 + L3', 'Rewritten code, 3 seeds')]:
        s.rect(x, 324, 346, 262, ('#FAFBFC', N[1]), layer=s.back)
        s.text(x + 173, 346, name, 13.5, True)
        s.box(x + 16, 358, 314, 98, PLAIN, [a, b, c])
        s.box(x + 16, 472, 314, 98, PLAIN,
              [d, e, 'Mean wF1 67.11, mF1 53.77' if x == 24 else 'Ensemble wF1 67.10, mF1 53.82'])
    s.write('overview.svg', title, 610)


E, I1, I2, I3 = (N, 'ECERC paper'), (L1, 'L1 Divyansh Dubey'), (L2, 'L2 Keshav Mishra'), (L3, 'L3 Lakshay Gupta')
TS = dict(name='Temperature scaling', body='One T fitted on validation, then softmax', c=L3)

stack('ecerc_baseline.svg', "ECERC as released by the authors", ['IEMOCAP: 1582-d', 'MELD: 300-d'], N, False,
      'Linear layer: 6 emotions on IEMOCAP, 7 on MELD', keys=[E],
      out=["Authors' checkpoints: IEMOCAP wF1 71.78, MELD wF1 66.46"])
stack('limitation1_wavlm_vad.svg', 'Limitation 1 alone: WavLM audio and VAD supervision', ['IEMOCAP: WavLM', '1024-d'], L1,
      False, 'Linear layer: 6 emotions, class-weighted loss',
      train=('VAD head, training only', ['Linear 1536 to 3', 'MSE to human ratings, weight 1.0'], 'L1'), keys=[E, I1],
      out=['Mean of 20 runs: wF1 71.13, mF1 69.73'])
stack('limitation2_llm_fusion.svg', 'Limitation 2 alone: ECERC with zero-shot LLM fusion', ['MELD: 300-d'], N, False,
      'Linear layer: 7 emotions, focal loss', keys=[E, I2],
      posts=[dict(name='Log-linear LLM fusion', body='0.8 log p(ECERC) + 0.2 log p(LLM)', c=L2,
                  side=['Qwen2.5-32B-Instruct', 'Zero-shot, never trained on', "MELD's labels"])],
      out=["On the authors' checkpoints: wF1 67.26, mF1 54.25"])
stack('limitation3_cacg.svg', 'Limitation 3 alone: ECERC with the CACG cause gate', ['IEMOCAP: 1582-d', 'MELD: 300-d'], N,
      True, 'Linear layer: 6 or 7 emotions',
      train=('Utility loss, training only', ['KL(q ‖ g) with one classifier', 'per cause. Optional.'], 'L3'),
      posts=[TS], keys=[E, I3], out=['Cause weights, calibrated confidence. No proven wF1 gain.'])
stack('approach_a_iemocap.svg', "Approach A, IEMOCAP: Divyansh Dubey's merge of L1 and L3", ['IEMOCAP: WavLM', '1024-d'], L1,
      True, 'Linear layer: 6 emotions, class-weighted loss',
      train=('VAD head, training only', ['Linear 1536 to 3', 'MSE to human ratings, weight 1.0'], 'L1'),
      posts=[TS, dict(name='Seed ensemble', body='Mean of 5 runs', c=L3)], keys=[E, I1, I3],
      out=['5-run ensemble: acc 72.18, wF1 72.43, mF1 71.14, ECE 2.12'])
stack('approach_a_meld.svg', "Approach A, MELD: Keshav Mishra's merge of L2 and L3", ['MELD: 300-d'], N, True,
      'Linear layer: 7 emotions, focal loss', keys=[E, I2, I3],
      posts=[dict(name='Log-linear LLM fusion', body='0.8 log p(ECERC) + 0.2 log p(LLM)', c=L2,
                  side=['Qwen2.5-32B-Instruct', 'Zero-shot, never trained on', "MELD's labels"]), TS],
      out=['Mean of 3 seeds: acc 68.17, wF1 67.11, mF1 53.77, ECE 3.18'])
unified()
overview()
