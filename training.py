import torch
import torchvision as tv
import numpy as np
import time

BATCH_SIZE = 256
NUM_EPOCHS = 5000
DUMP_EVERY = 100

def save_imgs(img_save_dir):
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    plt.figure(figsize=(4,4))
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0, hspace=0)
    images = generator(torch.rand((16, 128, 1, 1), device=device))
    for i in range(16):
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        fig = plt.imshow(images[i].detach().cpu().permute(1,2,0))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(img_save_dir,bbox_inches='tight',pad_inches=0) 

# Выбираем устройство для обучения
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Преобразование картинки
transforms = tv.transforms.Compose([
    tv.transforms.Resize(32),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.RandomRotation(10),
    tv.transforms.CenterCrop((32, 32)),
    tv.transforms.ToTensor()
])

dataset = tv.datasets.ImageFolder('anime_faces', transform=transforms)
loader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True, num_workers=4,
    pin_memory=True if use_cuda else False)

# Сетки
generator = torch.nn.Sequential(
    torch.nn.ConvTranspose2d(128, 64, (4,4), stride=2, padding=1),
    torch.nn.BatchNorm2d(64, momentum=0.5),
    torch.nn.LeakyReLU(0.2),
    
    torch.nn.ConvTranspose2d(64, 64, (4,4), stride=2, padding=1),
    torch.nn.BatchNorm2d(64, momentum=0.5),
    torch.nn.LeakyReLU(0.2),
    
    torch.nn.ConvTranspose2d(64, 32, (4,4), stride=2, padding=1),
    torch.nn.BatchNorm2d(32, momentum=0.5),
    torch.nn.LeakyReLU(0.2),
    
    torch.nn.ConvTranspose2d(32, 32, (4,4), stride=2, padding=1),
    torch.nn.BatchNorm2d(32, momentum=0.5),
    torch.nn.LeakyReLU(0.2),
    
    torch.nn.ConvTranspose2d(32, 3, (4,4), stride=2, padding=1),
    torch.nn.Tanh()
)

discriminator = torch.nn.Sequential(
    torch.nn.Conv2d(3, 32, kernel_size=(4,4), stride=2, padding=1),
    torch.nn.LeakyReLU(0.2),
    
    torch.nn.Conv2d(32, 32, kernel_size=(4,4), stride=2, padding=1),
    torch.nn.BatchNorm2d(32, momentum=0.5),
    torch.nn.LeakyReLU(0.2),
    
    torch.nn.Conv2d(32, 64, kernel_size=(4,4), stride=2, padding=1),
    torch.nn.BatchNorm2d(64, momentum=0.5),
    torch.nn.LeakyReLU(0.2),
    
    torch.nn.Conv2d(64, 64, kernel_size=(4,4), stride=2, padding=1),
    torch.nn.BatchNorm2d(64, momentum=0.5),
    torch.nn.LeakyReLU(0.2),
    
    torch.nn.AvgPool2d((2,2)),

    torch.nn.Flatten(),
    
    torch.nn.Linear(64, 1)
)

if use_cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()

discriminator_trainer = torch.optim.Adam(discriminator.parameters(), lr=0.015)
generator_trainer = torch.optim.Adam(generator.parameters(), lr=0.015)

loss = torch.nn.modules.BCEWithLogitsLoss()

print("Start training on device ", device)

for i in range(NUM_EPOCHS):
    loss_d, loss_g = 0., 0.
    start = time.time()
    for X, _ in loader:
        X = X.to(device)
        
        discriminator_trainer.zero_grad()
        generator_trainer.zero_grad()
        
        output = discriminator(X)
        l1 = loss(output, torch.ones((len(X),1), device=device))
        
        noise = torch.rand((BATCH_SIZE, 128, 1, 1), device=device)
        images = generator(noise)
        output = discriminator(images)
        l2 = loss(output, torch.zeros((BATCH_SIZE,1), device=device))
        l = (l1 + l2) / 2
        loss_d += l.item()
        l.backward()
        discriminator_trainer.step()
        
        discriminator_trainer.zero_grad()
        generator_trainer.zero_grad()
        noise = torch.rand((BATCH_SIZE, 128, 1, 1), device=device)
        images = generator(noise)
        output = discriminator(images)
        l = loss(output, torch.ones_like(output, device=device))
        loss_g += l.item()
        l.backward()
        generator_trainer.step()
        
    print("Epoch {}. Taked: {:.3f}, D: {:.3f}, G: {:.3f}".format(i, time.time() - start, loss_d, loss_g))
        
    if i % DUMP_EVERY == 0:
        torch.save(generator.state_dict(), "results/generator{}.params".format(i // DUMP_EVERY))
        torch.save(discriminator.state_dict(), "results/discriminator{}.params".format(i // DUMP_EVERY))
        
        save_imgs("images/img{}.png".format(i // DUMP_EVERY))
