import os, glob, random, zipfile, io
import numpy as np
from PIL import Image

# ---------- 1. UPLOADING THE FILES ----------
from google.colab import files
print(">>> Choose your reef_dataset.zip to upload...")
up = files.upload()
zip_name = list(up.keys())[0]
with zipfile.ZipFile(io.BytesIO(up[zip_name])) as z:
    z.extractall("data")

def find_dir(name):
    for root, _, _ in os.walk("data"):
        if os.path.basename(root).lower() == name: return root
    return None
IMG_DIR, MASK_DIR = find_dir("images"), find_dir("masks")
assert IMG_DIR and MASK_DIR, "images/ and masks/ not found in zip."
print("images:", IMG_DIR, "| masks:", MASK_DIR)

# ---------- 2. SETTTING UP THE MODEL ----------
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

IMG_SIZE, EPOCHS, BATCH, LR, SEED = 256, 100, 4, 1e-3, 42
N_TEST   = 3          # <<< number of COMPLETELY UNSEEN test slices
VAL_FRAC = 0.2        # fraction of the remainder used for validation
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------- 3. DATA ----------
class ReefData(Dataset):
    def __init__(s, imgs, masks, aug=False): s.i,s.m,s.aug=imgs,masks,aug
    def __len__(s): return len(s.i)
    def __getitem__(s,k):
        img=Image.open(s.i[k]).convert("L").resize((IMG_SIZE,IMG_SIZE))
        msk=Image.open(s.m[k]).convert("L").resize((IMG_SIZE,IMG_SIZE),Image.NEAREST)
        img=np.array(img,np.float32)/255.0
        msk=(np.array(msk,np.float32)>127).astype(np.float32)
        if s.aug and random.random()<0.5: img=img[:,::-1].copy(); msk=msk[:,::-1].copy()
        return torch.from_numpy(img)[None], torch.from_numpy(msk)[None]

# ---------- 4. USING U-NET ----------
class DC(nn.Module):
    def __init__(s,i,o):
        super().__init__(); s.n=nn.Sequential(
            nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True),
            nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True))
    def forward(s,x): return s.n(x)
class UNet(nn.Module):
    def __init__(s,b=32):
        super().__init__()
        s.d1,s.d2,s.d3,s.d4=DC(1,b),DC(b,b*2),DC(b*2,b*4),DC(b*4,b*8)
        s.p=nn.MaxPool2d(2); s.bo=DC(b*8,b*16)
        s.u4=nn.ConvTranspose2d(b*16,b*8,2,2); s.c4=DC(b*16,b*8)
        s.u3=nn.ConvTranspose2d(b*8,b*4,2,2);  s.c3=DC(b*8,b*4)
        s.u2=nn.ConvTranspose2d(b*4,b*2,2,2);  s.c2=DC(b*4,b*2)
        s.u1=nn.ConvTranspose2d(b*2,b,2,2);    s.c1=DC(b*2,b)
        s.o=nn.Conv2d(b,1,1)
    def forward(s,x):
        c1=s.d1(x);c2=s.d2(s.p(c1));c3=s.d3(s.p(c2));c4=s.d4(s.p(c3));bo=s.bo(s.p(c4))
        x=s.c4(torch.cat([s.u4(bo),c4],1));x=s.c3(torch.cat([s.u3(x),c3],1))
        x=s.c2(torch.cat([s.u2(x),c2],1));x=s.c1(torch.cat([s.u1(x),c1],1))
        return s.o(x)
class DiceBCE(nn.Module):
    def __init__(s): super().__init__(); s.b=nn.BCEWithLogitsLoss()
    def forward(s,lg,t):
        p=torch.sigmoid(lg);n=2*(p*t).sum((2,3))+1;d=p.sum((2,3))+t.sum((2,3))+1
        return s.b(lg,t)+(1-(n/d).mean())
@torch.no_grad()
def metrics(model,loader):
    model.eval();tp=fp=fn=0.0;D=[];I=[]
    for img,msk in loader:
        img,msk=img.to(device),msk.to(device)
        p=(torch.sigmoid(model(img))>0.5).float()
        tp+=((p==1)&(msk==1)).sum().item();fp+=((p==1)&(msk==0)).sum().item();fn+=((p==0)&(msk==1)).sum().item()
        it=(p*msk).sum((2,3));un=p.sum((2,3))+msk.sum((2,3))
        D+=((2*it+1)/(un+1)).cpu().tolist();I+=((it+1)/(un-it+1)).cpu().tolist()
    pr=tp/(tp+fp+1e-9);rc=tp/(tp+fn+1e-9);f1=2*pr*rc/(pr+rc+1e-9)
    return dict(dice=float(np.mean(D)),iou=float(np.mean(I)),precision=pr,recall=rc,f1=f1)

# ---------- 5. SPLITING THE DATA IN : train / val / TEST ----------
imgs=sorted(glob.glob(os.path.join(IMG_DIR,"*.png")))
imgs=[p for p in imgs if os.path.exists(os.path.join(MASK_DIR,os.path.basename(p)))]
masks=[os.path.join(MASK_DIR,os.path.basename(p)) for p in imgs]
print(f"{len(imgs)} labeled pairs found.")
assert len(imgs) >= N_TEST+5, "Not enough data for a test split."

idx=list(range(len(imgs))); random.shuffle(idx)
test_i = idx[:N_TEST]                       # <<< COMPLETELY UNSEEN, set aside first
rest   = idx[N_TEST:]
nv     = max(1,int(len(rest)*VAL_FRAC))
val_i  = rest[:nv]
tr_i   = rest[nv:]
print(f"TRAIN={len(tr_i)}  VAL={len(val_i)}  TEST(unseen)={len(test_i)}")
print("Test slices held out:", [os.path.basename(imgs[i]) for i in test_i])

mk=lambda I,a=False: ReefData([imgs[i] for i in I],[masks[i] for i in I],aug=a)
trl=DataLoader(mk(tr_i,True),BATCH,shuffle=True)
val=DataLoader(mk(val_i),BATCH)
tst=DataLoader(mk(test_i),BATCH)

# ---------- 6. TRAIN (checkpoint on VAL only) ----------
model=UNet().to(device); opt=torch.optim.Adam(model.parameters(),lr=LR); lossf=DiceBCE()
best=-1
for ep in range(EPOCHS):
    model.train()
    for img,msk in trl:
        img,msk=img.to(device),msk.to(device)
        opt.zero_grad(); lossf(model(img),msk).backward(); opt.step()
    mv=metrics(model,val)
    if (ep+1)%10==0 or ep==0: print(f"epoch {ep+1:3d}  val Dice {mv['dice']:.3f}")
    if mv['dice']>best: best=mv['dice']; torch.save(model.state_dict(),"best_model.pt"); best_val=mv

# ---------- 7. FINAL: score ON the UNTOUCHED test set ----------
model.load_state_dict(torch.load("best_model.pt"))
test_m = metrics(model, tst)      # first and only time the test set is used

report=f"""
REEF SEGMENTATION - RESULTS
============================================================
Labeled pairs            : {len(imgs)}
Split                    : train {len(tr_i)} / val {len(val_i)} / TEST(unseen) {len(test_i)}
Model                    : U-Net (baseline)
Loss                     : Dice + BCE   |  Epochs: {EPOCHS}

--- VALIDATION (used to pick best model) ---
Dice {best_val['dice']:.3f}  IoU {best_val['iou']:.3f}  Prec {best_val['precision']:.3f}  Rec {best_val['recall']:.3f}  F1 {best_val['f1']:.3f}

--- TEST  (COMPLETELY UNSEEN — the honest number) ---
Dice score (F1 overlap)  : {test_m['dice']:.3f}
IoU (Jaccard)            : {test_m['iou']:.3f}
Precision                : {test_m['precision']:.3f}
Recall                   : {test_m['recall']:.3f}
F1-score                 : {test_m['f1']:.3f}
(NOTE: test N={len(test_i)} is small, so this number is indicative, not precise.)
"""
print(report); open("metrics.txt","w").write(report)

# show the unseen test predictions
import matplotlib.pyplot as plt
model.eval()
with torch.no_grad():
    for img,msk in tst:
        pr=(torch.sigmoid(model(img.to(device)))>0.5).float().cpu()
        for j in range(img.shape[0]):
            fig,ax=plt.subplots(1,3,figsize=(11,4))
            ax[0].imshow(img[j,0],cmap="gray");ax[0].set_title("seismic (unseen)")
            ax[1].imshow(msk[j,0],cmap="gray");ax[1].set_title("your label")
            ax[2].imshow(pr[j,0],cmap="gray");ax[2].set_title("model prediction")
            for a in ax:a.axis("off"); plt.show()
print(">>> Done. The TEST numbers are your honest unseen-data result.")
