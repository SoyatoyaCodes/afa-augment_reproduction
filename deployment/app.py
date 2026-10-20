import math
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import streamlit as st
from PIL import Image
from torchvision import models

HERE = Path(__file__).parent
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


class DualBN2d(nn.Module):          # same as in training
    def __init__(self, n):
        super().__init__()
        self.bn_main = nn.BatchNorm2d(n)
        self.bn_aux = nn.BatchNorm2d(n)
        self.branch = "main"

    def forward(self, x):
        return self.bn_main(x) if self.branch == "main" else self.bn_aux(x)


def convert_to_dualbn(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, DualBN2d(child.num_features))
        else:
            convert_to_dualbn(child)
    return module


def set_branch(model, branch):
    for m in model.modules():
        if isinstance(m, DualBN2d):
            m.branch = branch


def base_resnet():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(512, 10)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)   # CIFAR stem (3x3)
    m.maxpool = nn.Identity()
    return m


@st.cache_resource
def load_models():
    b = base_resnet()
    b.load_state_dict(torch.load(HERE / "models" / "baseline_latest.pt", map_location="cpu"))
    a = convert_to_dualbn(base_resnet())
    a.load_state_dict(torch.load(HERE / "models" / "afa_3x3_80ep.pt", map_location="cpu"))
    set_branch(a, "main")           # test time: main branch only
    return b.eval(), a.eval()


def fourier_noise(x, strength, seed):
    g = torch.Generator().manual_seed(seed)
    B, C, H, W = x.shape
    f = 1 + 15 * torch.rand(B, C, 1, 1, generator=g)
    om = math.pi * torch.rand(B, C, 1, 1, generator=g)
    u = (torch.arange(H).float() / H).view(1, 1, H, 1)
    v = (torch.arange(W).float() / W).view(1, 1, 1, W)
    wave = torch.sin(2 * math.pi * f * (u * torch.cos(om) + v * torch.sin(om)) - math.pi / 4)
    wave = wave / wave.flatten(2).norm(dim=2).view(B, C, 1, 1)
    return (x + strength * wave).clamp(0, 1)


def gaussian_noise(x, std, seed):
    g = torch.Generator().manual_seed(seed)
    return (x + std * torch.randn(x.shape, generator=g)).clamp(0, 1)


def predict(model, x):
    with torch.no_grad():
        p = torch.softmax(model(x), 1)[0]
    vals, idx = p.topk(3)
    return [(CLASSES[i], float(v)) for v, i in zip(vals, idx)]


def show(x, caption):
    arr = (x.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    st.image(Image.fromarray(arr).resize((192, 192), Image.NEAREST), caption=caption)


st.title("Baseline vs AFA: CIFAR-10 classifier")
st.caption("ResNet-18, trained 80 epochs each. CIFAR-10-C robustness accuracy: "
           "baseline 74.67%, AFA 80.28%.")

baseline, afa = load_models()

samples = sorted((HERE / "sample_images").glob("*.png"))
source = st.radio("Image source", ["Sample image", "Upload"],
                  index=0 if samples else 1, horizontal=True)
img = None
if source == "Upload":
    f = st.file_uploader("Upload a JPG or PNG", type=["jpg", "jpeg", "png"])
    if f:
        img = Image.open(f).convert("RGB")
else:
    if samples:
        choice = st.selectbox("Sample", samples, format_func=lambda p: p.stem)
        img = Image.open(choice).convert("RGB")
    else:
        st.info("No sample images found. Use Upload instead.")

noise_type = st.selectbox("Noise", ["None", "Fourier-basis", "Gaussian"])
strength = st.slider("Noise strength", 0.0, 15.0 if noise_type == "Fourier-basis" else 0.5,
                     0.0, step=0.1 if noise_type == "Fourier-basis" else 0.01)
seed = st.number_input("Noise seed", 0, 9999, 0)

if img is not None:
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((32, 32), Image.BILINEAR)
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255).unsqueeze(0)
    x_noisy = x
    if noise_type == "Fourier-basis" and strength > 0:
        x_noisy = fourier_noise(x, strength, int(seed))
    elif noise_type == "Gaussian" and strength > 0:
        x_noisy = gaussian_noise(x, strength, int(seed))

    c1, c2 = st.columns(2)
    with c1: show(x, "Original (32x32)")
    with c2: show(x_noisy, "Model input")

    r1, r2 = st.columns(2)
    for col, name, m in [(r1, "Baseline", baseline), (r2, "AFA", afa)]:
        with col:
            st.subheader(name)
            for label, p in predict(m, x_noisy):
                st.write(f"{label}: {p:.1%}")
