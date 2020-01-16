import torch
import torchvision
import random
import os

from Generator import Generator
from Discriminator import Discriminator

if not os.path.exists("./samples"):
    os.makedirs("./samples")
if not os.path.exists("./models"):
    os.makedirs("./models")

torch.backends.cudnn.benchmark=True
device=torch.device("cuda:0") # GPU True

dataset = torchvision.datasets.LSUN(root="../data/", classes=["bedroom_train"],transform=torchvision.transforms.Compose([torchvision.transforms.Resize(64), torchvision.transforms.CenterCrop(64), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),]))

assert dataset
batch_size=64 # depending on the GPU
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

generator=Generator().to(device)
discriminator=Discriminator().to(device)
D_lr = 1e-5
G_lr = 2e-5
def weights_init(input):
    classname =input.__class__.__name__
    if classname.find('Conv') != -1:
        input.weight.data.normal_(0.0,0.02)
    elif classname.find('BatchNorm') !=-1:
        input.weight.data.normal_(1.0, 0.02)
        input.bias.data.fill_(0)

epochs=10
generator.apply(weights_init)

fixed_z = torch.FloatTensor(torch.randn(batch_size,100,1,1)).to(device)
real_label = torch.FloatTensor([1]).to(device)
fake_label = (real_label * -1).to(device)

optimD = torch.optim.RMSprop(discriminator.parameters(), lr=D_lr)
optimG = torch.optim.RMSprop(generator.parameters(), lr=G_lr)
def Clampping():
    # w <-clip(w,-c,c)
    for p in discriminator.parameters():
        p.data.clamp_(-0.01,0.01)

def train_discriminator(optimizer, real_data):
    discriminator.zero_grad()
    # real sample
    D_loss_real = discriminator(real_data.to(device))
    D_loss_real.backward(real_label)

    # fake sample
    with torch.no_grad():
        noise = torch.randn(batch_size,100,1,1).to(device)
    fake_data = generator(noise)
    D_loss_fake = discriminator(fake_data.detach())

    # w <- w+a*RMSProp(w,Gw)
    D_loss_fake.backward(fake_label)
    optimizer.step()
    return D_loss_real - D_loss_fake, fake_data

def train_generator(optimizer, fake_data):
    for p in discriminator.parameters():
        p.requries_grad = False
    generator.zero_grad()
    # 0 <- 0 - a*RMSProp(0,g0)
    D_loss_fake = discriminator(fake_data)
    D_loss_fake.backward(real_label)
    optimizer.step()
    return D_loss_fake # Estimating EM distance

print("Training start!")
f_loss = open("loss.txt","w")
# refer to a Algorithm of WGAN
for epoch in range(epochs):
    for i, data in enumerate(data_loader,0):
        for p in discriminator.parameters():
            p.requires_grad = True
        N_critic=0
        while N_critic < 5 and i < len(data_loader):
            j+=1
            D_loss, fake_data = train_discriminator(optimD,data[0])
            Clampping() # clampping parameters 
        G_loss = train_generator(optimG,fake_data)

        print("[%d] [%d] G loss : %.5f D_loss : %.5f " %(epoch, i, G_loss.item(), D_loss.item()))
        f_loss.write("%.5f/%.5f\n" %(G_loss,D_loss))
        f_loss.flush()

        # image save
        if (epoch*len(data_loader))+i % 1000 ==0:
            torchvision.utils.save_image(fake_data.detach(), "./samples/sample_%d.png" %(epoch*len(data_loader)+i), normalize=True)
print("Training Done!")









