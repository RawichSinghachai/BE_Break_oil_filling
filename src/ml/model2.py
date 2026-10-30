"""
model_v2.py — LSTM Autoencoder ที่ decoder รู้ 'phase' (stepnum) จึงตาม spike ตอนเริ่ม step 2 ได้

ต่างจาก model.py: decoder รับ z + ตำแหน่งเวลา + stepnum(phase) ของแต่ละแถว
stepnum เป็นสัญญาณควบคุมของเครื่อง (รู้ล่วงหน้าเสมอ) จึงใช้เป็น input ของ decoder ได้
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_feat, step_idx, hidden=64, latent=16, num_layers=1, dropout=0.0, max_len=120):
        """
        step_idx : index ของ lh_stepnum ใน FEATURES (ช่อง phase ที่จะป้อนให้ decoder)
        hidden/latent เพิ่มจากเดิม (32/8 -> 64/16) ให้มีที่เก็บรายละเอียด spike
        """
        super().__init__()
        self.n_feat, self.step_idx, self.max_len = n_feat, step_idx, max_len
        d = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(n_feat, hidden, num_layers, batch_first=True, dropout=d)
        self.to_latent = nn.Linear(hidden, latent)
        # decoder input = latent + ตำแหน่งเวลา(1) + stepnum(1)
        self.decoder = nn.LSTM(latent + 2, hidden, num_layers, batch_first=True, dropout=d)
        self.out = nn.Linear(hidden, n_feat)

    def encode(self, x, mask):
        lengths = mask.sum(1).cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.encoder(packed)
        return self.to_latent(h_n[-1])

    def decode(self, z, phase):
        """phase: [B, T, 1] = ค่า stepnum ของแต่ละแถว (บอก decoder ว่าอยู่ stage ไหน)"""
        B, T, _ = phase.shape
        pos = (torch.arange(T, device=z.device, dtype=z.dtype) / self.max_len).view(1, T, 1).expand(B, T, 1)
        z_rep = z.unsqueeze(1).expand(B, T, -1)
        d_in = torch.cat([z_rep, pos, phase], dim=-1)          # [B, T, latent+2]
        y, _ = self.decoder(d_in)
        return self.out(y)

    def forward(self, x, mask):
        z = self.encode(x, mask)
        phase = x[..., self.step_idx:self.step_idx + 1]        # ดึงช่อง stepnum ออกมาเป็น phase
        return self.decode(z, phase)


def masked_mse(recon, x, mask):
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
    B, T, F, STEP = 8, 120, 8, 0            # step_idx=0 เพราะ lh_stepnum เป็น feature แรก
    lengths = torch.randint(45, T + 1, (B,))
    mask = torch.arange(T)[None, :] < lengths[:, None]
    x = torch.randn(B, T, F) * mask.unsqueeze(-1)
    model = LSTMAutoencoder(n_feat=F, step_idx=STEP, max_len=T)
    recon = model(x, mask)
    loss = masked_mse(recon, x, mask); loss.backward()
    print("recon", tuple(recon.shape), "loss", float(loss),
          "params", sum(p.numel() for p in model.parameters()))