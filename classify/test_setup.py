import torch
import cv2

print("✅ PyTorch:", torch.__version__)
print("✅ CUDA available:", torch.cuda.is_available())
print("✅ GPU:", torch.cuda.get_device_name(0))
print("✅ OpenCV:", cv2.__version__)
print("\n🎉 Environment is ready for PPE Detection!")