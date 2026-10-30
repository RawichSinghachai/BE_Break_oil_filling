"""
model3.py — LSTM Autoencoder รุ่นปรับปรุงจาก model2.py

สิ่งที่เปลี่ยน (ดูเหตุผลในแชท):
  1) decoder รู้ "ตำแหน่งใน phase" ละเอียดขึ้น: ช่วงที่เท่าไร (one-hot ของลำดับช่วง step),
     เวลาตั้งแต่เริ่มช่วง, สัดส่วนในช่วง, เวลาที่เหลือของช่วง, เวลาที่เหลือของทั้งรอบ
     -> แยก idle ตอนต้นกับตอนท้ายรอบออกจากกันได้ และวาง spike/drop ตอนท้ายรอบได้ตรงขึ้น
  2) encoder เป็น bidirectional + attention/mean pooling (เดิมใช้ h_n ตัวสุดท้าย ทำให้ลืมช่วงต้นรอบ
     เช่น ระดับ volume ก่อนเติม และ spike ของ vacuum ตอนเริ่ม)
  3) decoder เป็น bidirectional + pack ตามความยาวจริง (ขอบขาขึ้น/ลงคมขึ้น ไม่หน่วง, ไม่ ringing ติดลบ)
  4) loss แบบถ่วงน้ำหนัก feature + Huber (เน้น 3 channel ที่ใช้ตัดสิน, ไม่เสียแรงกับ stepnum)
  5) denoising (noise_std) เป็นตัวเลือก ช่วย generalize เมื่อมีข้อมูลน้อย (415 คัน)

interface เดิมยังใช้ได้: LSTMAutoencoder(n_feat, step_idx, max_len=...), model(x, mask),
masked_mse, anomaly_score
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import MAX_LEN


# ---------------------------------------------------------------------------
# features บอก phase ให้ decoder (คำนวณจาก stepnum เท่านั้น ไม่ได้ใช้ค่าสัญญาณที่จะทาย)
# ---------------------------------------------------------------------------
def phase_features(step, mask, max_len, n_seg=8):
    """
    step : [B, T] ค่า stepnum (scaled แล้วก็ได้ ใช้แค่ตำแหน่งที่ค่าเปลี่ยน)
    mask : [B, T] bool
    return [B, T, n_seg + 5]
        one-hot ลำดับช่วง (0=ช่วงแรก, 1=ช่วงถัดไป ...)  -> idle ต้นรอบ ≠ idle ท้ายรอบ
        pos      : t / max_len
        rev      : เวลาที่เหลือถึงจบรอบ / max_len
        t_in     : เวลาตั้งแต่เริ่มช่วงปัจจุบัน / max_len
        frac     : สัดส่วนที่ผ่านไปในช่วงปัจจุบัน (0..1)
        t_left   : เวลาที่เหลือของช่วงปัจจุบัน / max_len
    """
    B, T = step.shape
    dev, dt = step.device, step.dtype
    idx = torch.arange(T, device=dev).unsqueeze(0).expand(B, T)               # long

    change = torch.zeros_like(mask)
    change[:, 1:] = (step[:, 1:] - step[:, :-1]).abs() > 1e-4
    change = change & mask                                                    # ไม่นับรอยต่อกับ padding

    seg = change.long().cumsum(1).clamp(max=n_seg - 1)                        # ลำดับช่วงของแต่ละแถว
    m = mask.unsqueeze(-1).to(dt)
    seg_oh = F.one_hot(seg, n_seg).to(dt) * m                                 # [B, T, n_seg]

    start = torch.where(change | (idx == 0), idx, torch.zeros_like(idx))
    start = torch.cummax(start, dim=1).values                                 # index ที่ช่วงปัจจุบันเริ่ม
    t_in = (idx - start).to(dt)

    seg_len = seg_oh.sum(1)                                                   # [B, n_seg]
    seg_len_t = (seg_oh * seg_len.unsqueeze(1)).sum(-1).clamp(min=1)          # ความยาวช่วงของแต่ละแถว
    frac = t_in / seg_len_t
    t_left = (seg_len_t - 1 - t_in).clamp(min=0) / max_len

    L = mask.sum(1, keepdim=True)
    pos = idx.to(dt) / max_len
    rev = (L - 1 - idx).clamp(min=0).to(dt) / max_len

    extra = torch.stack([pos, rev, t_in / max_len, frac, t_left], dim=-1)    # [B, T, 5]
    return torch.cat([seg_oh, extra], dim=-1) * m


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_feat, step_idx, hidden=64, latent=24, num_layers=2, dropout=0.1,
                 max_len=120, n_seg=8, noise_std=0.0):
        """
        hidden/latent/num_layers : ความจุของโมเดล (ลอง latent 16/24/32)
        n_seg     : จำนวนช่วง step สูงสุดที่แยก (idle->vacuum->fill->idle = 4, เผื่อไว้ 8)
        noise_std : >0 = denoising AE ตอน train (ลอง 0.02-0.05 บนข้อมูลที่ scale แล้ว)
        """
        super().__init__()
        self.n_feat, self.step_idx, self.max_len = n_feat, step_idx, max_len
        self.n_seg, self.noise_std = n_seg, noise_std
        d = dropout if num_layers > 1 else 0.0

        self.encoder = nn.LSTM(n_feat, hidden, num_layers, batch_first=True,
                               dropout=d, bidirectional=True)
        self.attn = nn.Linear(2 * hidden, 1)
        self.to_latent = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.GELU(),
                                       nn.Linear(hidden, latent))

        cond_dim = n_seg + 5
        self.decoder = nn.LSTM(latent + cond_dim, hidden, num_layers, batch_first=True,
                               dropout=d, bidirectional=True)
        self.out = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_feat))

    def _run_lstm(self, lstm, seq, mask):
        lengths = mask.sum(1).cpu()
        packed = pack_padded_sequence(seq, lengths, batch_first=True, enforce_sorted=False)
        h, _ = lstm(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=seq.size(1))
        return h                                                               # [B, T, 2H]

    def encode(self, x, mask):
        h = self._run_lstm(self.encoder, x, mask)
        m = mask.unsqueeze(-1)
        a = self.attn(h).masked_fill(~m, float("-inf")).softmax(dim=1)        # attention pooling
        att = (a * h).sum(1)
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)                         # mean pooling
        return self.to_latent(torch.cat([att, mean], dim=-1))

    def decode(self, z, cond, mask):
        z_rep = z.unsqueeze(1).expand(-1, cond.size(1), -1)
        y = self._run_lstm(self.decoder, torch.cat([z_rep, cond], dim=-1), mask)
        return self.out(y)

    def forward(self, x, mask):
        cond = phase_features(x[..., self.step_idx], mask, self.max_len, self.n_seg)
        x_in = x
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            noise[..., self.step_idx] = 0.0                                   # ไม่ใส่ noise ให้ stepnum
            x_in = x + noise * mask.unsqueeze(-1)
        z = self.encode(x_in, mask)
        return self.decode(z, cond, mask)


# ---------------------------------------------------------------------------
# loss / score
# ---------------------------------------------------------------------------
def make_feat_weights(features, focus=None, zero=("lh_stepnum",), default=1.0):
    """
    สร้าง weight ต่อ feature เช่น
      make_feat_weights(FEATURES, focus={"lh_fillingpressure": 3, "lh_fillingvolume": 3, "lh_vacuumgun": 3})
    feature ใน zero ได้ weight 0 (stepnum ถูกป้อนให้ decoder อยู่แล้ว ไม่ต้องให้ทาย)
    """
    focus = focus or {}
    w = [0.0 if f in zero else float(focus.get(f, default)) for f in features]
    return torch.tensor(w)


def masked_loss(recon, x, mask, feat_w=None, kind="huber", delta=1.0):
    """kind: 'huber' (แนะนำ) | 'mse' | 'l1'  —  feat_w: tensor [F] หรือ None"""
    if kind == "huber":
        e = F.huber_loss(recon, x, reduction="none", delta=delta)
    elif kind == "l1":
        e = (recon - x).abs()
    else:
        e = (recon - x) ** 2
    e = e[mask]                                                                # [N, F]
    if feat_w is None:
        return e.mean()
    w = feat_w.to(e.device, e.dtype)
    return (e * w).sum(-1).mean() / w.sum()


def masked_mse(recon, x, mask):                                               # คงไว้ให้โค้ดเดิมใช้ได้
    return ((recon - x) ** 2)[mask].mean()


@torch.no_grad()
def anomaly_score(model, x, mask, channel=None, mode="mean"):
    model.eval()
    recon = model(x, mask)
    err = (recon - x) ** 2
    err = err.mean(-1) if channel is None else err[..., channel]
    err = err.masked_fill(~mask, 0.0)
    if mode == "max":
        return err.max(dim=1).values
    return err.sum(1) / mask.sum(1)


if __name__ == "__main__":
    B, T, F_, STEP = 8, MAX_LEN, 8, 0
    lengths = torch.randint(45, T + 1, (B,))
    mask = torch.arange(T)[None, :] < lengths[:, None]
    # stepnum จำลอง: idle -> vacuum -> fill -> idle
    step = torch.zeros(B, T)
    for i, L in enumerate(lengths.tolist()):
        step[i, 3:15], step[i, 15:L - 10] = 1, 2
    x = torch.randn(B, T, F_)
    x[..., STEP] = step
    x = x * mask.unsqueeze(-1)

    model = LSTMAutoencoder(n_feat=F_, step_idx=STEP, max_len=T, noise_std=0.03)
    w = make_feat_weights([f"f{i}" for i in range(F_)], zero=("f0",))
    recon = model(x, mask)
    loss = masked_loss(recon, x, mask, w)
    loss.backward()
    print("cond", tuple(phase_features(x[..., STEP], mask, T).shape),
          "recon", tuple(recon.shape), "loss", float(loss),
          "params", sum(p.numel() for p in model.parameters()))