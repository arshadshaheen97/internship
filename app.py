import streamlit as st
import torch
from torch import nn
import numpy as np
from PIL import Image
import io

# Define Generator model
class Generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=28*28):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + 10, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat((noise, labels), dim=1)
        return self.model(x)

# Load generator
generator = Generator()
generator.load_state_dict(torch.load("generator.pth", map_location=torch.device('cpu')))
generator.eval()

# UI
st.title("MNIST Handwritten Digit Generator")
digit = st.number_input("Select a digit (0–9)", min_value=0, max_value=9, step=1)

if st.button("Generate"):
    label = torch.zeros((5, 10))
    label[range(5), digit] = 1
    noise = torch.randn(5, 100)

    with torch.no_grad():
        images = generator(noise, label).view(-1, 28, 28)

        st.subheader(f"Generated Images for Digit {digit}")
        cols = st.columns(5)
        for i, img in enumerate(images):
            img_np = ((img + 1) * 127.5).clamp(0, 255).byte().numpy()
            img_pil = Image.fromarray(img_np)
            cols[i].image(img_pil.resize((128, 128)), caption=f"Sample {i+1}")
