import torch

import custom_model

# Number of classes in the dataset
num_classes = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_deeplabv3, input_size = custom_model.initialize_model(num_classes, keep_feature_extract=True, use_pretrained=False)

state_dict = torch.load("training_output_Skydiver_dataset_person/best_DeepLabV3_Skydiver.pth", map_location=device)

model_deeplabv3 = model_deeplabv3.to(device)
model_deeplabv3.load_state_dict(state_dict)
model_deeplabv3.eval()

model_deeplabv3wrapper = custom_model.DeepLabV3Wrapper(model_deeplabv3)

dummy_input = torch.rand(1, 3, input_size, input_size).to(device)
traced_script_module = torch.jit.trace(model_deeplabv3wrapper, dummy_input)
traced_script_module.save("training_output_Skydiver_dataset_person/best_deeplabv3_skydiver.pt")
