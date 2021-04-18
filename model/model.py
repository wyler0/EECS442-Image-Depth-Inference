import os
import json
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as T
import random
import numpy as np
import torch.nn as nn

class ImageDepthPredModel(nn.Module):
    def __init__(self):
        super().__init__()

        ## Encoders ##
        # import pretrained models
        densenet169 = models.densenet169(pretrained=True)

        # remove the classification and batch normalization part of the model
        self.encoder_densenet = torch.nn.Sequential(*list(densenet169.features)[:-1])

        # freeze parameters of the pretrained encoder
        for param in self.encoder_densenet.parameters():
            param.requires_grad = False

        self.activation = {}
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output.detach()
            return hook

        # Register hooks to save intermediate values
        self.encoder_densenet[0].register_forward_hook(get_activation('conv1'))
        self.encoder_densenet[3].register_forward_hook(get_activation('pool1'))
        self.encoder_densenet[5][3].register_forward_hook(get_activation('pool2'))
        self.encoder_densenet[7][3].register_forward_hook(get_activation('pool3'))
        ####


        ## decoder
        self.conv0 = nn.Conv2d(in_channels=1664, out_channels=1664, kernel_size=1)

        self.up1 = nn.Upsample(scale_factor=2)
        self.up1_conv_a = nn.Conv2d(in_channels=1920, out_channels=832, kernel_size=3, stride=1, padding=1)
        self.up1_conv_b = nn.Conv2d(in_channels=832, out_channels=832, kernel_size=3, stride=1, padding=1)
        self.up1_relu = nn.LeakyReLU(negative_slope=0.01)

        self.up2 = nn.Upsample(scale_factor=2)
        self.up2_conv_a = nn.Conv2d(in_channels=960, out_channels=416, kernel_size=3, stride=1, padding=1)
        self.up2_conv_b = nn.Conv2d(in_channels=416, out_channels=416, kernel_size=3, stride=1, padding=1)
        self.up2_relu = nn.LeakyReLU(negative_slope=0.01)

        self.up3 = nn.Upsample(scale_factor=2)
        self.up3_conv_a = nn.Conv2d(in_channels=480, out_channels=208, kernel_size=3, stride=1, padding=1)
        self.up3_conv_b = nn.Conv2d(in_channels=208, out_channels=208, kernel_size=3, stride=1, padding=1)
        self.up3_relu = nn.LeakyReLU(negative_slope=0.01)

        self.up4 = nn.Upsample(scale_factor=2)
        self.up4_conv_a = nn.Conv2d(in_channels=272, out_channels=104, kernel_size=3, stride=1, padding=1)
        self.up4_conv_b = nn.Conv2d(in_channels=104, out_channels=104, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=104, out_channels=1, kernel_size=3, stride=1, padding=1)
        ##

    # def init_weights(self):
    #     # TODO: initialize the parameters for
    #     #       [self.fc1, self.fc2, self.fc3, self.deconv]
    #     for fc in [self.fc1, self.fc2, self.fc3]:
    #         C_in = fc.weight.size(1)
    #         nn.init.normal_(fc.weight, 0.0, 0.1 / sqrt(C_in))
    #         nn.init.constant_(fc.bias, 0.0)

    #     for deconv in [self.deconv]:
    #         C_in = deconv.weight.size(1)
    #         nn.init.normal_(deconv.weight, 0.0, 0.01)
    #         nn.init.constant_(deconv.bias, 0.0)
        #

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encoder(self, x):
        encoded_dense169 = self.encoder_densenet(x)
        return encoded_dense169

    def decoder(self, encoded):
        conv0 = self.conv0(encoded)

        up1 = self.up1(conv0)
        concat1 = torch.cat((up1, self.activation['pool3']), dim=1)
        up1_conva = self.up1_conv_a(concat1)
        up1_convb = self.up1_conv_b(up1_conva)
        up1_res = self.up1_relu(up1_convb)

        up2 = self.up2(up1_res)
        concat2 = torch.cat((up2, self.activation['pool2']), dim=1)
        up2_conva = self.up2_conv_a(concat2)
        up2_convb = self.up2_conv_b(up2_conva)
        up2_res = self.up1_relu(up2_convb)

        up3 = self.up3(up2_res)
        concat3 = torch.cat((up3, self.activation['pool1']), dim=1)
        up3_conva = self.up3_conv_a(concat3)
        up3_convb = self.up3_conv_b(up3_conva)
        up3_res = self.up1_relu(up3_convb)

        up4 = self.up4(up3_res)
        concat4 = torch.cat((up4, self.activation['conv1']), dim=1)
        up4_conva = self.up4_conv_a(concat4)
        up4_convb = self.up4_conv_b(up4_conva)

        decoded = self.conv3(up4_convb)
        return decoded