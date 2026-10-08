"""ECERC with pluggable cause gating. With gate='baseline' this is numerically identical to the
official release and loads the released checkpoints (parameter names are unchanged).

forward() returns a dict so the other two extensions can hook in:
  out['log_prob']  (N, C)      flattened over valid utterances -> class-balanced / focal loss (Limitation 2)
  out['fused']     (N, 4*3h)   representation fed to the classifier -> contrastive regularizer (Limitation 1)
  out['gate']      dict         CACG internals -> gate loss (Limitation 3)
"""
import torch
import torch.nn.functional as F
from torch import nn

from layers import Encoder, ModalFilterV2, PositionalEncoding
from cacg import CauseGate, build_descriptor


class ECERC(nn.Module):
    def __init__(self, d_t, d_a, d_v, base_layer=1, hidden_size=128, n_classes=6,
                 pos_scale=1.0, dropout1=0.5, dropout2=0.5,
                 gate='baseline', gate_kwargs=None, fix_empty_attention=False):
        super().__init__()
        self.hidden_size, self.d_t, self.d_a, self.d_v = hidden_size, d_t, d_a, d_v
        self.fix_empty_attention = fix_empty_attention
        input_size = d_t + d_a + d_v
        self.position_enc_evi = PositionalEncoding(input_size, n_position=200, scale=pos_scale)
        self.dropout1 = nn.Dropout(p=dropout1)
        self.dropout2 = nn.Dropout(p=dropout2)
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)  # unused in the release, kept for checkpoint compat

        self.modal_filtering = ModalFilterV2(d_t, d_a, d_v, hidden_size)
        D = hidden_size * 3
        self.emotion_encoding = Encoder(n_layer=base_layer, n_head=8, d_model=D, d_k=64, d_v=64)
        self.proj_eve = nn.Linear(d_t, hidden_size)
        self.event_encoding = Encoder(n_layer=base_layer, n_head=8, d_model=hidden_size, d_k=64, d_v=64)
        self.self_contagion_attention = Encoder(n_layer=base_layer, n_head=8, d_model=D, d_k=64, d_v=64)
        self.cross_emotion_attention = Encoder(n_layer=base_layer, n_head=8, d_model=D, d_k=64, d_v=64)
        self.self_event_attention = Encoder(n_layer=base_layer, n_head=8, d_model=hidden_size, d_k=64, d_v=64)
        self.cross_event_attention = Encoder(n_layer=base_layer, n_head=8, d_model=hidden_size, d_k=64, d_v=64)
        self.gate_reset = nn.Linear(D, D)
        self.gate_reset2 = nn.Linear(D, D)
        self.smax_fc = nn.Linear(D * 4, n_classes)

        self.gate_mode = gate
        self.cause_gate = CauseGate(D, n_classes, mode=gate, **(gate_kwargs or {})) if gate != 'baseline' else None

    def forward(self, U_e, U_s, qmask, umask, seq_lengths):
        """U_e: (L,B,d_t+d_a+d_v) evidence; U_s: (L,B,d_t) semantic/event; qmask: (L,B,P); umask: (B,L)."""
        U_e_, U_s_, qmask_ = U_e.transpose(0, 1), U_s.transpose(0, 1), qmask.transpose(0, 1)
        B, L = U_e_.size(0), U_e_.size(1)
        dev = U_e.device
        U_e_ = self.position_enc_evi(U_e_)
        U_s_ = self.position_enc_evi(U_s_)

        # masks (identical to the release, made device-agnostic)
        imask = torch.eye(L, device=dev).unsqueeze(0).bool()
        submask = (1 - torch.triu(torch.ones((1, L, L), device=dev), diagonal=1)).bool()
        padmask = umask.unsqueeze(-2).bool()
        P = qmask_.size(2)
        weights = 2 ** torch.arange(P - 1, -1, -1, device=dev).float()
        spk = torch.matmul(qmask_, weights).int()  # (B,L) speaker id, as in the release
        same = spk.unsqueeze(1) == spk.unsqueeze(2)  # (B,L,L) [i,j]: speaker(i)==speaker(j)
        smask = submask & padmask & same
        cmask = submask & padmask & ~same
        mask = submask & padmask
        cont_mask = smask & ~imask

        # number of available history items per cause (used for availability + descriptor)
        n_self = cont_mask.sum(-1).float()
        n_cross = cmask.sum(-1).float()

        if self.fix_empty_attention:
            # rows with no valid key fall back to attending to the utterance itself
            # (release: uniform attention over the whole padded dialogue incl. future turns)
            cont_mask = cont_mask | (imask & (n_self == 0).unsqueeze(-1))
            cmask = cmask | (imask & (n_cross == 0).unsqueeze(-1))

        evidence_enc, modal_gates = self.modal_filtering(
            U_e_[:, :, :self.d_t], U_e_[:, :, self.d_t:self.d_t + self.d_a], U_e_[:, :, self.d_t + self.d_a:])
        evidence_enc = self.dropout1(evidence_enc)
        emotion_enc = self.emotion_encoding(evidence_enc, evidence_enc, evidence_enc, "self", mask)
        U_s_ = self.proj_eve(U_s_)
        event_enc = self.event_encoding(U_s_, U_s_, U_s_, "self", mask)

        h = self.hidden_size
        R_self_event = self.self_event_attention(evidence_enc[:, :, :h], event_enc, event_enc, "cross", imask)
        R_self_event = torch.cat([R_self_event, evidence_enc[:, :, h:]], dim=-1)
        R_cross_event = self.cross_event_attention(evidence_enc[:, :, :h], event_enc, event_enc, "cross", cmask)
        R_cross_event = torch.cat([R_cross_event, evidence_enc[:, :, h:]], dim=-1)
        R_contagion = self.self_contagion_attention(evidence_enc, emotion_enc, emotion_enc, "cross", cont_mask)
        R_cross_emotion = self.cross_emotion_attention(evidence_enc, emotion_enc, emotion_enc, "cross", cmask)

        # Feature Gating (release): element-wise sigmoid gate per cause, then concatenation
        R_lis = [R_contagion, R_cross_emotion, R_self_event, R_cross_event]
        wR_lis = [torch.sigmoid(self.gate_reset(R)) * R if idx < 2 else torch.sigmoid(self.gate_reset2(R)) * R
                  for idx, R in enumerate(R_lis)]

        gate_info = None
        desc = build_descriptor(modal_gates, n_self, n_cross, wR_lis)
        if self.cause_gate is None:
            R = torch.cat(wR_lis, dim=-1)
        else:
            avail = torch.stack([n_self > 0, n_cross > 0, torch.ones_like(n_self, dtype=torch.bool), n_cross > 0], -1)
            R, gate_info = self.cause_gate(wR_lis, desc, avail)

        hidden = self.smax_fc(self.dropout2(R))
        valid = umask.bool()
        out = {
            'log_prob': F.log_softmax(hidden, -1)[valid],
            'fused': R[valid],
            'gate': gate_info,
            'valid': valid,
            'n_ctx': (n_self + n_cross)[valid],
            'desc': desc[valid],
        }
        return out
