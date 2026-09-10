import sys
import torch
import torchvision
import sklearn
import pandas
import numpy
import PIL
import matplotlib
import thop

print("=== AMBIENTE ===")
print("Python:", sys.version)

print("PyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Pandas:", pandas.__version__)
print("NumPy:", numpy.__version__)
print("Pillow:", PIL.__version__)
print("Matplotlib:", matplotlib.__version__)
print("THOP:", thop.__version__)

print("\n=== DISPOSITIVO ===")
print("CUDA disponível:", torch.cuda.is_available())

print("\n=== TESTE PYTORCH ===")
x = torch.rand(3, 3)
print(x)

print("\nAmbiente funcionando!")