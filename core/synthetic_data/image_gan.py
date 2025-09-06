import torch
import torch.nn as nn
from torchvision.utils import save_image
import os
from utils.logging import get_logger

logger = get_logger(__name__)

class ImageGANGenerator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(ImageGANGenerator, self).__init__()
        self.img_shape = img_shape
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(torch.prod(torch.tensor(self.img_shape)))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

class ImageGAN:
    """
    A simple GAN for generating synthetic images.
    """
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.generator = ImageGANGenerator(self.latent_dim, self.img_shape)
        self.discriminator = Discriminator(self.img_shape)
        self.adversarial_loss = nn.BCELoss()
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)
        self.discriminator.to(self.device)

    def train_step(self, real_imgs):
        """
        Performs one training step for the GAN.
        """
        # Configure input
        real_imgs = real_imgs.to(self.device)
        valid = torch.full((real_imgs.size(0), 1), 1.0, device=self.device)
        fake = torch.full((real_imgs.size(0), 1), 0.0, device=self.device)

        # -----------------
        #  Train Generator
        # -----------------
        self.optimizer_g.zero_grad()
        z = torch.randn(real_imgs.size(0), self.latent_dim, device=self.device)
        gen_imgs = self.generator(z)
        g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)
        g_loss.backward()
        self.optimizer_g.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        self.optimizer_d.zero_grad()
        real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
        fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.optimizer_d.step()

        return g_loss.item(), d_loss.item()
        
    def generate_images(self, num_images: int, output_dir: str):
        """
        Generates and saves a specified number of synthetic images.
        """
        logger.info(f"Generating {num_images} synthetic images...")
        self.generator.eval()
        os.makedirs(output_dir, exist_ok=True)
        
        z = torch.randn(num_images, self.latent_dim, device=self.device)
        with torch.no_grad():
            gen_imgs = self.generator(z)
        
        for i in range(num_images):
            img_path = os.path.join(output_dir, f"gen_img_{i}.png")
            save_image(gen_imgs[i], img_path, normalize=True)
        
        logger.info(f"Saved {num_images} generated images to {output_dir}")
        self.generator.train()
        return [os.path.join(output_dir, f"gen_img_{i}.png") for i in range(num_images)]