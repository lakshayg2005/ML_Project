"""Draws the merged system's architecture. Usage: python make_architecture.py  (writes architecture.png / .pdf here)"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

plt.rcParams.update({'font.family': 'Arial', 'mathtext.fontset': 'stixsans', 'pdf.fonttype': 42})
BASE = dict(fc='#EEF0F3', ec='#5B6573')
BAND = dict(fc='#F8F9FA', ec='#D5D9DF')
L3 = dict(fc='#DCEBFA', ec='#1F6FB5')
L2 = dict(fc='#FCE8D5', ec='#C8640F')
INK, MUTED = '#1E2530', '#5B6573'

def box(x, y, w, h, style, title=None, sub=None, ts=10.5, ss=9, lw=1.2, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.8', lw=lw, ls=ls, **style))
    cy = y + h / 2
    if title and sub:
        ax.text(x + w / 2, cy + h * .17, title, ha='center', va='center', fontsize=ts, fontweight='bold', color=INK)
        ax.text(x + w / 2, cy - h * .2, sub, ha='center', va='center', fontsize=ss, color=MUTED, linespacing=1.3)
    elif title:
        ax.text(x + w / 2, cy, title, ha='center', va='center', fontsize=ts, fontweight='bold', color=INK)


def arrow(p, q, color=MUTED, lw=1.3, ls='-', rad=0.):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=11, lw=lw, ls=ls, color=color, shrinkA=0,
                                 shrinkB=0, connectionstyle=f'arc3,rad={rad}'))


def stage(y, h, label, style=BAND, x=1.8, ly=None):
    ax.add_patch(FancyBboxPatch((0.5, y), 73, h, boxstyle='round,pad=0,rounding_size=1', lw=.8, **style))
    color = MUTED if style is BAND else style['ec']
    ax.text(x, ly or y + h - 1.4, label, ha='left', va='center', fontsize=9.5, fontweight='bold', color=color)


here = os.path.dirname(os.path.abspath(__file__))
for with_llm, name in [(True, 'architecture'), (False, 'architecture_ecerc_cacg')]:
    fig, ax = plt.subplots(figsize=(10, 12.6) if with_llm else (7.8, 11.4))
    ax.set(xlim=(0, 100 if with_llm else 74), ylim=(0 if with_llm else 6, 126))
    ax.axis('off')

    # Inputs
    Y = 117.5
    for x, w, t, s in [(8, 13.5, 'Text', '1024-d'), (23, 13.5, 'Audio slot', '300-d'), (38, 13.5, 'Visual', '342-d'),
                       (56, 16, 'Event / semantic', '1024-d')]:
        box(x, Y, w, 6.5, BASE, t, s)
    ax.text(1.8, Y + 3.25, 'Input', fontsize=9.5, fontweight='bold', color=MUTED, va='center')

    # 1 Evidence Gating
    stage(102, 12, '1  Evidence Gating', x=52.6, ly=108)
    box(8, 103.3, 43.5, 8, BASE, 'Modality gates', 'project each to 128-d, sigmoid gate, concatenate')
    for x in (14.75, 29.75, 44.75):
        arrow((x, Y), (x, 111.3))

    # 2 Cause Encoding
    stage(85, 14.5, '2  Cause Encoding', x=25)
    box(8, 86.5, 29, 9, BASE, 'Emotion encoder', 'Transformer on evidence, 384-d')
    box(43, 86.5, 29, 9, BASE, 'Event encoder', 'projection to 128-d, Transformer')
    arrow((22.5, 103.3), (22.5, 95.5))
    arrow((70, Y), (70, 95.5))
    ax.text(24, 100.5, 'evidence 384-d', fontsize=8.5, color=MUTED, va='center')

    # 3 Cause attentions
    stage(66, 16.5, '3  Cause Attentions', x=31.5, ly=81.2)
    causes = [(3, 'Self-contagion', 'past emotions,\nsame speaker'), (21, 'Cross-emotion', 'past emotions,\nother speakers'),
              (39, 'Self-event', "the utterance's\nown event"), (57, 'Cross-event', 'past events,\nother speakers')]
    for x, t, s in causes:
        box(x, 67.3, 16, 10.7, BASE, t, s, ts=10, ss=8.5)
    for x in (14, 29, 50, 65):
        arrow((x, 86.5), (x, 78))

    ax.plot([8, 4.5, 4.5, 60], [107, 107, 79.7, 79.7], color=MUTED, lw=1.2, ls='--')
    for x in (6, 24, 42, 60):
        arrow((x, 79.7), (x, 78), lw=1.1)
    ax.text(6.2, 91, 'evidence as query', rotation=90, ha='center', va='center', fontsize=8.5, color=MUTED)

    # 4 Feature gating
    stage(53.5, 10, '4  Feature Gating', x=12.5)
    box(8, 54.6, 64, 5.6, BASE, r'Sigmoid gate per cause (each 384-d):   $f_k \leftarrow \sigma(W f_k)\ \odot\ f_k$', ts=10)
    for x, lab in [(11, 'f1'), (29, 'f2'), (47, 'f3'), (65, 'f4')]:
        arrow((x, 67.3), (x, 60.2))

    # 5 CACG
    stage(29.5, 21.5, '5  Confidence-Aware Cause Gate (CACG, Limitation 3)', L3)
    box(3, 38.5, 22, 8.6, dict(fc='white', ec=L3['ec']), 'Quality descriptor h', 'modality gates (3), context\ncounts (3), cause cosines (6)', ts=9.5, ss=8)
    box(28, 38.5, 16, 8.6, dict(fc='white', ec=L3['ec']), 'τ network', 'τ in [0.25, 4]', ts=9.5, ss=8.5)
    box(47, 38.5, 25, 8.6, dict(fc='white', ec=L3['ec']), 'Softmax gate', r'$g=\mathrm{softmax}(W_g[f_1;..;f_4]/\tau)$' + '\nno-history causes masked', ts=9.5, ss=8.5)
    box(14, 30.7, 58, 5.6, dict(fc='white', ec=L3['ec']), r'Re-weight and concatenate:   $[4g_1f_1;\ 4g_2f_2;\ 4g_3f_3;\ 4g_4f_4]$', ts=10)
    arrow((25, 42.8), (28, 42.8), color=L3['ec'])
    arrow((44, 42.8), (47, 42.8), color=L3['ec'])
    arrow((59.5, 54.6), (59.5, 47.1), color=L3['ec'])
    arrow((59.5, 38.5), (59.5, 36.3), color=L3['ec'])

    # 6 Classifier
    stage(16.5, 10.5, '6  Classifier')
    box(14, 17.6, 58, 5.8, BASE, 'Dropout 0.2, Linear, log-softmax   →   ' + r'$\log p_{\mathrm{ECERC}}$', ts=10)
    arrow((43, 30.7), (43, 23.4))
    ax.text(44, 27.9, 'fused 1536-d', fontsize=8.5, color=MUTED, va='center')

    if with_llm:
        # LLM branch
        ax.add_patch(FancyBboxPatch((75.5, 16.5), 24, 107.5, boxstyle='round,pad=0,rounding_size=1', lw=.8, **dict(L2, fc='#FFF8F1')))
        ax.text(87.5, 121.8, 'LLM branch (Limitation 2)', ha='center', va='center', fontsize=9.5, fontweight='bold', color=L2['ec'])
        ax.text(87.5, 119.6, 'trained separately from ECERC', ha='center', va='center', fontsize=8.5, color=MUTED)
        white_l2 = dict(fc='white', ec=L2['ec'])
        box(77, 103, 21, 11, white_l2, 'Dialogue so far', 'text + generic\nspeaker labels', ts=9.5, ss=8.5)
        box(77, 79, 21, 14, white_l2, 'Qwen2.5-32B', 'zero-shot, 4-bit;\nnever trained on\nMELD labels', ts=9.5, ss=8.5)
        box(77, 53, 21, 15, white_l2, 'Prior matching', 'label-free:\n− mean dev prediction\n+ train class prior', ts=9.5, ss=8.5)
        box(77, 19, 21, 11, L2, r'$\log\ \tilde{p}_{\mathrm{LLM}}$', '7-class log-probs', ts=11, ss=8.5)
        for a, b in [(103, 93), (79, 68), (53, 30)]:
            arrow((87.5, a), (87.5, b), color=L2['ec'])

        # 7 Fusion
        ax.add_patch(FancyBboxPatch((0.5, 5.5), 99, 8.5, boxstyle='round,pad=0,rounding_size=1', lw=.8, **L2))
        ax.text(1.8, 12.6, '7  Late fusion (Limitation 2)', ha='left', va='center', fontsize=9.5, fontweight='bold', color=L2['ec'])
        ax.text(50, 8.9, r'$\log p_{\mathrm{final}} = (1-w)\ \log p_{\mathrm{ECERC}} + w\ \log \tilde{p}_{\mathrm{LLM}}$, '
                + '   w = 0.2 (chosen on dev)', ha='center', va='center', fontsize=10.5, color=INK)
        arrow((43, 17.6), (43, 14), color=L2['ec'])
        arrow((87.5, 19), (87.5, 14), color=L2['ec'])

        # Output
        arrow((50, 5.5), (50, 3.7), color=INK)
        ax.text(50, 2.6, 'Predicted emotion: neutral, surprise, fear, sadness, joy, disgust, anger',
                ha='center', va='center', fontsize=10, fontweight='bold', color=INK)

    else:
        arrow((43, 17.6), (43, 13.6), color=INK)
        ax.text(43, 12.2, 'Predicted emotion: neutral, surprise, fear, sadness, joy, disgust, anger', ha='center',
                va='center', fontsize=10, fontweight='bold', color=INK)
    handles = [Patch(fc=BASE['fc'], ec=BASE['ec'], label="Authors' ECERC (unchanged)"),
               Patch(fc=L3['fc'], ec=L3['ec'], label='Added by Limitation 3')]
    if with_llm:
        handles.append(Patch(fc=L2['fc'], ec=L2['ec'], label='Added by Limitation 2'))
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, -.022 if with_llm else -.012), ncol=3,
              frameon=False, fontsize=9.5)
    for ext_ in ('png', 'pdf'):
        fig.savefig(os.path.join(here, f'{name}.{ext_}'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
