# Consolidated Imports
from imports import *


# Additional Imports
from dataset import SkinCancer

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default='../data/', help="number of epochs of training")
parser.add_argument("--csv_files", type=str, default='../csv/', help="number of epochs of training")
parser.add_argument("--n_epochs", type=int, default=5, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=7, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator(opt.n_classes, opt.latent_dim, opt.img_size, opt.channels)
discriminator = Discriminator(opt.img_size, opt.channels, opt.n_classes)

# CUDA
if device == 'cuda':
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Directories and Dataloader
data_dir = opt.data+'HAM10k/HAM10000_images/'
train_data = SkinCancer(data_dir, opt.csv_files+'train.csv', transform=None)
dataset_size = len(train_data)    
test_data = SkinCancer(data_dir, opt.csv_files+'test.csv',transform=None)
classes=np.unique(train_data.classes)
dataloader = torch.utils.data.DataLoader(train_data, batch_size=16)
unnormalize =transforms.Normalize((-0.5 / 0.5), (1.0 / 0.5))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if device == 'cuda' else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "../data/HAM10k/GAN_Generated/%d.png" % batches_done, nrow=n_row, normalize=True)

def sample_imgs(gen_imgs,gen_labels,epoch):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    imgs = [i.detach().cpu().squeeze().permute(1,2,0) for i in gen_imgs]
    imgs = [unnormalize(i) for i in imgs]
    gen_lab = [int(i.detach().cpu().numpy().tolist()) for i in gen_labels]
    titles = [train_data.class_id[i] for i in gen_lab]
    # fig = plt.figure(figsize=(12, 8))
    f, axes = plt.subplots(4,4,figsize=(12,10))
    
    for idx,img in enumerate(imgs):
        i = idx % 4
        j = idx // 4
        axes[i,j].imshow(img);
        plt.subplots_adjust(wspace=.3, hspace=0.3)
        # a = fig.add_subplot(4, 4, i+1)
        # imgplot = plt.plot(imgs[i])
        # a.axis("off")
        # axes[i,j].set_title(titles[idx], fontsize=10, fontweight="bold")
    plt.savefig(f'../save/epoch_{epoch+1}_generated_batch.jpg', bbox_inches='tight')


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        if i < 165:
            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            # real_imgs = Variable(imgs.type(FloatTensor))
            # labels = Variable(labels.type(LongTensor))
            
            real_imgs = imgs
            labels = torch.as_tensor(labels)
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity, pred_label = discriminator(gen_imgs)
            g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            real_pred, real_aux = discriminator(real_imgs)
            d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

            # Loss for fake images
            fake_pred, fake_aux = discriminator(gen_imgs.detach())
            d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            # Calculate discriminator accuracy
            pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
            gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
            d_acc = np.mean(np.argmax(pred, axis=1) == gt)

            d_loss.backward()
            optimizer_D.step()

            print(f'Epoch {epoch}/{opt.n_epochs} Batch {i}/{len(dataloader)} D loss: {d_loss.item()} acc: {100 * d_acc} G loss: {g_loss.item()}')
            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                sample_imgs(n_row=10, batches_done=batches_done)
