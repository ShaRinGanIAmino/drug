
import argparse
import torch
import torch.nn as nn
import numpy as np
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Lipinski
import streamlit as st
from rdkit.Chem.Draw import MolToImage

# --- 1. Fetch & preprocess SMILES ---
def fetch_smiles(target_id='CHEMBL4497824', limit=1000):
    activity = new_client.activity
    acts = activity.filter(target_chembl_id=target_id, standard_type='IC50').only(['canonical_smiles'])
    smiles = [a['canonical_smiles'] for a in acts][:limit]
    return [s for s in smiles if Chem.MolFromSmiles(s)]

# --- 2. Vocabulary & Encoding ---
def build_vocab(smiles_list):
    chars = sorted({c for s in smiles_list for c in s})
    stoi = {c: i+1 for i, c in enumerate(chars)}
    stoi['<pad>'] = 0
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos

def encode(s, stoi, maxlen=100):
    arr = [stoi[c] for c in s] + [0]*(maxlen - len(s))
    return torch.tensor(arr[:maxlen])

# --- 3. VAE Model ---
class VAE(nn.Module):
    def __init__(self, vocab_size, emb=64, hid=128, lat=64):
        super().__init__()
        self.E = nn.Embedding(vocab_size, emb)
        self.enc = nn.GRU(emb, hid, batch_first=True)
        self.mu = nn.Linear(hid, lat)
        self.logv = nn.Linear(hid, lat)
        self.dec = nn.GRU(emb+lat, hid, batch_first=True)
        self.out = nn.Linear(hid, vocab_size)

    def forward(self, x):
        emb = self.E(x)
        _, h = self.enc(emb)
        m, lv = self.mu(h[-1]), self.logv(h[-1])
        z = m + (0.5*lv).exp() * torch.randn_like(m)
        zc = z.unsqueeze(1).repeat(1, x.size(1), 1)
        dec_in = torch.cat([emb, zc], -1)
        o, _ = self.dec(dec_in)
        return self.out(o), m, lv

# --- 4. Training & Generation ---
def train(smiles, stoi, epochs=10, bs=32):
    data = torch.stack([encode(s, stoi) for s in smiles])
    loader = torch.utils.data.DataLoader(data, bs, shuffle=True)
    model = VAE(len(stoi)).to('cpu')
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    ce = nn.CrossEntropyLoss(ignore_index=0)
    for ep in range(epochs):
        total = 0
        for batch in loader:
            opt.zero_grad()
            logits, mu, lv = model(batch)
            rec = ce(logits.transpose(1,2), batch)
            kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()
            (rec+kl).backward()
            opt.step()
            total += rec.item()
        print(f"Epoch {ep} rec={total/len(loader):.3f}")
    return model

def decode(idx_seq, itos):
    return ''.join(itos[i] for i in idx_seq if i != 0)

def generate(model, itos, n=5, maxlen=100):
    zs = torch.randn(n, model.mu.out_features)
    sample = torch.zeros(n, maxlen, dtype=torch.long)
    logits, _, _ = model(sample)
    idx = logits.argmax(-1)
    return [decode(i.tolist(), itos) for i in idx]

# --- 5. Lipinski Filter ---
def lipinski_pass(smi):
    m = Chem.MolFromSmiles(smi)
    return sum([
        Lipinski.NHOHCount(m) > 5,
        Lipinski.NOCount(m) > 10,
        Chem.Crippen.MolLogP(m) > 5,
        Chem.Descriptors.MolWt(m) > 500
    ]) <= 1

# --- CLI & Streamlit ---
def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['train', 'gen', 'serve'])
    args = p.parse_args()

    if args.action == 'train':
        smi = fetch_smiles()
        stoi, itos = build_vocab(smi)
        model = train(smi, stoi)
        torch.save({'model': model.state_dict(), 'stoi': stoi, 'itos': itos}, 'vae.pth')

    elif args.action == 'gen':
        ckpt = torch.load('vae.pth')
        model = VAE(len(ckpt['stoi']))
        model.load_state_dict(ckpt['model'])
        out = generate(model, ckpt['itos'], n=20)
        for s in out:
            if Chem.MolFromSmiles(s) and lipinski_pass(s):
                print(s)

    else:  # serve
        st.title("Compact Molecule VAE Demo")
        
        # --- Here insert your note only in Streamlit ---
        st.markdown(
            """
            <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
                <h4 style='color: #333;'>🔬 Model Information</h4>
                <p>This model generates novel drug-like molecules targeting the SARS-CoV-2 main protease (Mpro).</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("Train & Generate"):
            with st.spinner("Fetching data..."):
                smi = fetch_smiles()
            stoi, itos = build_vocab(smi)
            model = train(smi, stoi, epochs=5)
            st.success("Trained!")
            examples = generate(model, itos, n=10)
            for sm in examples:
                if Chem.MolFromSmiles(sm) and lipinski_pass(sm):
                    img = MolToImage(Chem.MolFromSmiles(sm))
                    st.image(img)
                    st.write(sm)

if __name__ == '__main__':
    main()
