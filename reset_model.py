from models import SimpleCNN_STFT_FRAMESIZE_1024

model = SimpleCNN_STFT_FRAMESIZE_1024()

for layer in model.children():
   if hasattr(layer, 'reset_parameters'):
       layer.reset_parameters()