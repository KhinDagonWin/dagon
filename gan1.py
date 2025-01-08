import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            # 64x64 -> 32x32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 32x32 -> 16x16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 16x16 -> 8x8
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64x64 -> 128x128
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output values between -1 and 1 for RGB images
        )

    def forward(self, x):
        x = self.main(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x64 -> 32x32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32 -> 16x16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16 -> 8x8
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Flatten to a single output for real/fake classification
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return x
    


# Initialize models
G_A2B = Generator().to(device)
G_B2A = Generator().to(device)
D_A = Discriminator().to(device)
D_B = Discriminator().to(device)

# Optimizers
optimizer_G = optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_A = optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss functions
criterion_gan = nn.MSELoss().to(device)
criterion_cycle = nn.L1Loss().to(device)

# Data loading and transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

apple_dataset = datasets.ImageFolder(root='D:/program/python/gai/CP', transform=transform)
orange_dataset = datasets.ImageFolder(root='D:/program/python/gai/CP', transform=transform)

apple_loader = DataLoader(apple_dataset, batch_size=1, shuffle=True)
orange_loader = DataLoader(orange_dataset, batch_size=1, shuffle=True)

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    for i, (apple_data, orange_data) in enumerate(zip(apple_loader, orange_loader)):
        real_A = apple_data[0].to(device)
        real_B = orange_data[0].to(device)

        # ---------------------
        # Train Generators
        # ---------------------
        optimizer_G.zero_grad()

        # GAN loss G_A2B
        fake_B = G_A2B(real_A)
        pred_fake_B = D_B(fake_B)
        loss_GAN_A2B = criterion_gan(pred_fake_B, torch.ones_like(pred_fake_B).to(device))

        # GAN loss G_B2A
        fake_A = G_B2A(real_B)
        pred_fake_A = D_A(fake_A)
        loss_GAN_B2A = criterion_gan(pred_fake_A, torch.ones_like(pred_fake_A).to(device))

        # Cycle consistency loss
        recov_A = G_B2A(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)

        recov_B = G_A2B(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)

        # Total generator loss
        loss_G = loss_GAN_A2B + loss_GAN_B2A + 10 * (loss_cycle_A + loss_cycle_B)
        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        # Train Discriminators
        # ---------------------
        optimizer_D_A.zero_grad()
        optimizer_D_B.zero_grad()

        # Discriminator A
        pred_real_A = D_A(real_A)
        loss_D_real_A = criterion_gan(pred_real_A, torch.ones_like(pred_real_A).to(device))

        pred_fake_A = D_A(fake_A.detach())
        loss_D_fake_A = criterion_gan(pred_fake_A, torch.zeros_like(pred_fake_A).to(device))

        loss_D_A = (loss_D_real_A + loss_D_fake_A) / 2
        loss_D_A.backward()
        optimizer_D_A.step()

        # Discriminator B
        pred_real_B = D_B(real_B)
        loss_D_real_B = criterion_gan(pred_real_B, torch.ones_like(pred_real_B).to(device))

        pred_fake_B = D_B(fake_B.detach())
        loss_D_fake_B = criterion_gan(pred_fake_B, torch.zeros_like(pred_fake_B).to(device))

        loss_D_B = (loss_D_real_B + loss_D_fake_B) / 2
        loss_D_B.backward()
        optimizer_D_B.step()

    print(f"Epoch [{epoch}/{num_epochs}], Loss G: {loss_G.item()}, Loss D_A: {loss_D_A.item()}, Loss D_B: {loss_D_B.item()}")

    # Save sample generated images
    if epoch % 10 == 0:
        save_image(fake_B, f"output/fake_cattle_{epoch}.jpg", normalize=True)
        save_image(fake_A, f"output/fake_pig_{epoch}.jpg", normalize=True)

# Save final models
torch.save(G_A2B.state_dict(), "G_A2B.pth")
torch.save(G_B2A.state_dict(), "G_B2A.pth")
torch.save(D_A.state_dict(), "D_A.pth")
torch.save(D_B.state_dict(), "D_B.pth")
