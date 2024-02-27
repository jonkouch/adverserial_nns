# visualize the saved adversarial attack
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# # Load the model
# model_path = 'models/cifar10/Linf/student_model_best_699.pt'
# model = torch.load(model_path, map_location=torch.device('cpu'))

# # Print the model architecture
# print(model)


path = 'results/pert_upgd_robust.pt'
pert = torch.load(path).detach().cpu().numpy()
pert = np.transpose(pert, (1, 2, 0))

# load example from the dataset and add the perturbation
# path = 'results/clean_img.png'
# # load with values between 0 and 1
# img = Image.open(path)
# img = np.array(img)
# img = img / 255
# img = img + pert
# img = np.clip(img, 0, 1)


# # Convert back to PIL Image to save
# img = np.transpose(img, (2, 0, 1))  # Change back to CHW format for PyTorch
# img = torch.tensor(img)
# img = img.numpy()  # Convert tensor to numpy array
# img = np.transpose(img, (1, 2, 0))  # Convert to HWC for PIL
# img = Image.fromarray((img * 255).astype(np.uint8))  # Convert to uint8 and back to PIL Image

# # Save the image
# img.save('results/upgd_example_robust.png')



pert = pert/(8/255)
pert = np.clip(pert, 0, 1)
pert = Image.fromarray((pert * 255).astype(np.uint8))
# save the image
pert.save('results/pert_upgd_robust_vis.png')



