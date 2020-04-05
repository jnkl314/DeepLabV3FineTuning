import torch
import numpy as np
from torchvision import transforms
import cv2
from PIL import Image

import custom_model

# Number of classes in the dataset
num_classes = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model, input_size = custom_model.initialize_model(num_classes, keep_feature_extract=True, use_pretrained=False)

state_dict = torch.load("training_output_Skydiver_dataset_final/best_DeepLabV3_Skydiver.pth", map_location=device)

model = model.to(device)
model.load_state_dict(state_dict)
model.eval()

transforms_image =  transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

for idx in range(1, 3000, 25):

    image = Image.open(f"/tmp/pycharm_project_782/03.03.20_saut_4/{idx:06}.png")

    image_np = np.asarray(image)
    # image_np = cv2.resize(image_np, 0.5, 0.5, cv2.INTER_CUBIC)
    width = int(image_np.shape[1] * 0.3)
    height = int(image_np.shape[0] * 0.3)
    dim = (width, height)
    image_np = cv2.resize(image_np, dim, interpolation=cv2.INTER_AREA)

    image = Image.fromarray(image_np)
    image = transforms_image(image)
    image = image.unsqueeze(0)

    image = image.to(device)

    outputs = model(image)["out"]

    _, preds = torch.max(outputs, 1)

    preds = preds.to("cpu")

    preds_np = preds.squeeze(0).cpu().numpy().astype(np.uint8)

    print(preds_np.shape)
    print(image_np.shape)
    # preds_np = cv2.cvtColor(preds_np, cv2.COLOR_GRAY2BGR)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    preds_np_color = cv2.applyColorMap(preds_np * 50, cv2.COLORMAP_HSV)

    cv2.imwrite(f"./results/{idx:04}_segmentation.png", preds_np_color)
    cv2.imwrite(f"./results/{idx:04}_image.png", image_np)


