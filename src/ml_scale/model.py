"""
model.py — LSTM Autoencoder สำหรับตรวจความผิดปกติของรอบเติมน้ำมันเบรก (pgnum = 20)

รับ input ตรงกับ DataLoader ในโค้ดเตรียมข้อมูลของคุณ:
    x    : [B, T, F]  float32  (padded ด้วย 0 ท้ายแถว)
    mask : [B, T]     bool     (True = แถวจริง, False = padding)

ผลลัพธ์ forward: recon [B, T, F] = รอบที่โมเดลสร้างกลับ
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_feat, hidden=32, latent=8, num_layers=1, dropout=0.0, max_len=120):
        """
        n_feat     : จำนวน feature (F) = len(FEATURES)  ต้องตรงกับ input
        hidden     : ขนาด hidden state ของ LSTM
        latent     : ขนาด vector คอขวด (ยิ่งเล็ก ยิ่งบีบแรง)
        num_layers : จำนวนชั้น LSTM (เริ่มที่ 1 พอ)
        dropout    : ใช้ได้เมื่อ num_layers > 1
        max_len    : ความยาวสูงสุดของรอบ (T) ใช้ normalize ตำแหน่งเวลา
        """
        super().__init__()
        self.n_feat = n_feat
        self.max_len = max_len
        d = dropout if num_layers > 1 else 0.0

        # ---------- ENCODER : [B,T,F] -> latent [B, latent] ----------
        self.encoder = nn.LSTM(n_feat, hidden, num_layers, batch_first=True, dropout=d)
        self.to_latent = nn.Linear(hidden, latent)

        # ---------- DECODER : latent [B, latent] -> recon [B,T,F] ----------
        # input ของ decoder = latent (ซ้ำทุกก้าว) + ตำแหน่งเวลา 1 ค่า
        self.decoder = nn.LSTM(latent + 1, hidden, num_layers, batch_first=True, dropout=d)
        self.out = nn.Linear(hidden, n_feat)

    # ----- encode: อ่านทั้งรอบ -> ใจความ -----
    def encode(self, x, mask):
        lengths = mask.sum(1).cpu()                                  # ความยาวจริงต่อคัน จาก mask
        packed = pack_padded_sequence(x, lengths, batch_first=True,  # ให้ LSTM ข้าม padding
                                      enforce_sorted=False)
        _, (h_n, _) = self.encoder(packed)                          # h_n: [num_layers, B, hidden]
        return self.to_latent(h_n[-1])                              # ชั้นบนสุด -> latent [B, latent]

    # ----- decode: ใจความ -> สร้างรอบกลับ -----
    def decode(self, z, T):
        B = z.size(0)
        pos = torch.arange(T, device=z.device, dtype=z.dtype) / self.max_len   # 0..~1 บอกตำแหน่งในรอบ
        z_rep = z.unsqueeze(1).expand(B, T, -1)                      # ซ้ำ latent ทุกก้าว [B,T,latent]
        pos_rep = pos.view(1, T, 1).expand(B, T, 1)                  # [B,T,1]
        d_in = torch.cat([z_rep, pos_rep], dim=-1)                   # [B,T,latent+1]
        y, _ = self.decoder(d_in)                                    # [B,T,hidden]
        return self.out(y)                                          # [B,T,F]

    def forward(self, x, mask):
        z = self.encode(x, mask)
        recon = self.decode(z, x.size(1))
        return recon                                               # [B,T,F]


# ---------- loss: คิดเฉพาะแถวจริง ----------
def masked_mse(recon, x, mask):
    """MSE เฉพาะแถวที่เป็นข้อมูลจริง (padding ไม่นับ)"""
    return ((recon - x) ** 2)[mask].mean()


# ---------- คะแนนผิดปกติต่อคัน ----------
@torch.no_grad()
def anomaly_score(model, x, mask, channel=None, mode="mean"):
    """
    คืน error ต่อคัน [B]
      channel=None       -> เฉลี่ยทุก feature
      channel=i          -> ดูเฉพาะ feature ที่ i (เช่น fillingpressure)
      mode="mean"        -> error เฉลี่ยตลอดรอบ
      mode="max"         -> error พีคสูงสุด (ใช้จับความผิดปกติแบบแหลมสั้น)
    """
    model.eval()
    recon = model(x, mask)
    err = (recon - x) ** 2                                          # [B,T,F]
    err = err.mean(-1) if channel is None else err[..., channel]    # [B,T]
    err = err.masked_fill(~mask, 0.0)                               # padding = 0
    if mode == "max":
        return err.max(dim=1).values                               # [B]
    return err.sum(1) / mask.sum(1)                                # [B]  เฉลี่ยเฉพาะแถวจริง

