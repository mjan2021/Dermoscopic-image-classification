
from imports import *

class Discriminator(nn.Module):
    def __init__(self, img_size=32, channels=1, n_classes=7):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.channels = channels
        self.n_classes = n_classes
        

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(self.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = self.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, self.n_classes), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label

class Generator(nn.Module):
    def __init__(self, n_classes=7, latent_dim=100, img_size=32, channels=1):
        super(Generator, self).__init__()
        self.n_classes = n_classes
        self.latent_dim= latent_dim
        self.img_size = img_size
        self.channels = channels
        

        self.label_emb = nn.Embedding(self.n_classes, self.latent_dim)
        self.init_size = self.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class cnn():
    pass


def efficientnet():
    model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT')
    old_fc = model.classifier.__getitem__(-1)
    new_fc = nn.Linear(in_features=old_fc.in_features, out_features= 7, bias=True)
    model.classifier.__setitem__(-1 , new_fc)
    return model

def resnet():
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    old_fc = model.fc
    new_fc = nn.Linear(in_features=old_fc.in_features, out_features= 7, bias=True)
    model.fc = new_fc
    
    return model


def vit():
    model = models.vit_b_16(weights='ViT_B_16_Weights.DEFAULT')
    
    old_fc = model.heads.__getitem__(-1)
    new_fc = nn.Linear(in_features=old_fc.in_features, out_features= 7, bias=True)
    model.heads.__setitem__(-1 , new_fc)

    return model


def convnext():
    model = models.convnext_base(weights='ConvNeXt_Base_Weights.DEFAULT')
    old_fc = model.classifier.__getitem__(-1)
    new_fc = nn.Linear(in_features=old_fc.in_features, out_features= 7, bias=True)
    model.classifier.__setitem__(-1 , new_fc)

    return model

def alexnet():
    model = models.alexnet(pretrained=True)
    old_fc = model.classifier.__getitem__(-1)
    new_fc = nn.Linear(in_features=old_fc.in_features, out_features= 7, bias=True)
    model.classifier.__setitem__(-1 , new_fc)
    
    return model


def resnext():
    model = models.resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
    old_fc = model.fc
    new_fc = nn.Linear(in_features=old_fc.in_features, out_features= 7, bias=True)
    model.fc = new_fc
    
    return model


