import torch
import segmentation_models_pytorch as smp

sd = torch.load('best_cable_unet.pth', map_location='cpu')

for enc in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'mobilenet_v2']:
    try:
        model = smp.Unet(encoder_name=enc, encoder_weights=None, classes=1)
        model.load_state_dict(sd)
        print("SUCCESS:", enc)
        break
    except Exception as e:
        # print("Failed", enc)
        pass
else:
    print("No matching encoder found!")
