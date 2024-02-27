import torch
import torch.nn.functional as F  # Correct import for functional
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.models import resnet18  # Assuming you're using ResNet18 for student model
from robustbench.utils import load_model
from tqdm.auto import tqdm
import numpy as np

# Custom Transform: Add Epsilon-Bounded Noise
class AddEpsilonNoise(object):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, img):
        noise = torch.empty_like(img).uniform_(-self.epsilon, self.epsilon)
        return img.add(noise).clamp(0, 1)  # Ensure tensor values are in [0, 1] after noise addition

# Parameters
batch_size = 64*16
num_workers = 16
pin_memory = False
epsilon = 0.03
use_kl_divergence = True
epochs = 1000

# Transforms
transform_train = transforms.Compose([
    transforms.ToTensor(),
    AddEpsilonNoise(epsilon=epsilon),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# Datasets
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Data Loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

# Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teacher_model = load_model(model_name='Wong2020Fast', dataset='cifar10', threat_model='Linf').to(device)
student_model = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf').to(device)

if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    student_model = torch.nn.DataParallel(student_model)

# Criterion and Loss Adjustment for KL Divergence
if use_kl_divergence:
    criterion = torch.nn.KLDivLoss(reduction='batchmean')
else:
    criterion = torch.nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)

# Scheduler and Regularization
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Initialize variables to track the best test loss and its epoch
best_test_loss = float('inf')
best_epoch = 0

test_losses, train_losses = [], []

# Assuming the total size of train_loader and test_loader is known
total_steps = epochs * (len(train_loader) + len(test_loader))
progress_bar = tqdm(total=total_steps, desc='Initial setup')
average_test_loss = float('inf')
for epoch in range(epochs):
    student_model.train()
    train_loss = 0.0
    num_batches_train = 0

    for inputs, _ in train_loader:
        inputs = inputs.to(device)
        optimizer.zero_grad()

        with autocast():
            outputs = student_model(inputs)
            if use_kl_divergence:
                outputs = F.log_softmax(outputs, dim=1)
                teacher_outputs = teacher_model(inputs)
                targets = F.softmax(teacher_outputs, dim=1)
            else:
                targets = teacher_model(inputs)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        num_batches_train += 1

        # Update progress bar with real-time train loss
        average_train_loss = train_loss / num_batches_train
        progress_bar.set_description(
            f'Epoch {epoch + 1}/{100}, Train Loss: {average_train_loss:.4f}, Test Loss: {average_test_loss:.4f}, Best Test Loss: {best_test_loss:.4f} @ Epoch {best_epoch}')
        progress_bar.update()

    # Testing phase
    student_model.eval()
    test_loss = 0.0
    num_batches_test = 0

    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = student_model(inputs)
            if use_kl_divergence:
                outputs = F.log_softmax(outputs, dim=1)
                teacher_outputs = teacher_model(inputs)
                targets = F.softmax(teacher_outputs, dim=1)
            else:
                targets = teacher_model(inputs)
            loss = criterion(outputs, targets)
        test_loss += loss.item()
        num_batches_test += 1

        # Update progress bar with real-time test loss
        average_test_loss = test_loss / num_batches_test
        progress_bar.set_description(
            f'Epoch {epoch + 1}/{100}, Train Loss: {average_train_loss:.4f}, Test Loss: {average_test_loss:.4f}, Best Test Loss: {best_test_loss:.4f} @ Epoch {best_epoch}')
        progress_bar.update()

    # Update best test loss
    if average_test_loss < best_test_loss:
        best_test_loss = average_test_loss
        best_epoch = epoch + 1
        torch.save(student_model.state_dict(), f'student_model_best_{best_epoch}.pt')  # Save the best model

    # Final update to progress bar at the end of epoch to include best test loss info
    progress_bar.set_description(
        f'Epoch {epoch + 1}/{100}, Train Loss: {average_train_loss:.4f}, Test Loss: {average_test_loss:.4f}, Best Test Loss: {best_test_loss:.4f} @ Epoch {best_epoch}')
    test_losses.append(average_test_loss)
    train_losses.append(average_train_loss)


    scheduler.step()

progress_bar.close()

# Save Model
torch.save(student_model.state_dict(), 'final_student_model.pt')
print("Model training complete and saved.")



#save the losses as a txt file
np.savetxt('test_losses.txt', test_losses, delimiter=',')
np.savetxt('train_losses.txt', train_losses, delimiter=',')
