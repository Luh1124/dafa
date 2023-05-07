import argparse
from models import EFE_6 as EFE
from models import AFE, CKD, HPE_EDE, MFE
from models import Generator_FPN as Generator
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import imageio
import os
from skimage import io, img_as_float32
from utils import transform_kp, transform_kp_with_new_pose
from logger import Visualizer
import torchvision
import math
from utils import apply_imagenet_normalization
import glob
import cv2
import tqdm

class Hopenet(nn.Module):
    # Hopenet with 3 output layers for yaw, pitch and roll
    # Predicts Euler angles by binning and regression with the expected value
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(Hopenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)

        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)
        self.idx_tensor = torch.FloatTensor(list(range(num_bins))).unsqueeze(0).cuda()
        self.n_bins = num_bins
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        real_yaw = self.fc_yaw(x)
        real_pitch = self.fc_pitch(x)
        real_roll = self.fc_roll(x)
        real_yaw = torch.softmax(real_yaw, dim=1)
        real_pitch = torch.softmax(real_pitch, dim=1)
        real_roll = torch.softmax(real_roll, dim=1)
        real_yaw = (real_yaw * self.idx_tensor).sum(dim=1)
        real_pitch = (real_pitch * self.idx_tensor).sum(dim=1)
        real_roll = (real_roll * self.idx_tensor).sum(dim=1)
        real_yaw = (real_yaw - self.n_bins // 2) * 3 * np.pi / 180
        real_pitch = (real_pitch - self.n_bins // 2) * 3 * np.pi / 180
        real_roll = (real_roll - self.n_bins // 2) * 3 * np.pi / 180

        return real_yaw, real_pitch, real_roll

def edit_exp(args):
    hp = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66).cuda()
    hp.load_state_dict(torch.load("hopenet_robust_alpha1.pkl", map_location=torch.device("cpu")))
    for parameter in hp.parameters():
        parameter.requires_grad = False
    g_models = {"efe": EFE(), "afe": AFE(), "ckd": CKD(), "hpe_ede": HPE_EDE(), "mfe": MFE(), "generator": Generator()}
    ckp_path = os.path.join(args.ckp_dir, "%s-checkpoint.pth.tar" % str(args.ckp).zfill(8))
    checkpoint = torch.load(ckp_path, map_location=torch.device("cpu"))
    for k, v in g_models.items():
        v.cuda()
        v.load_state_dict(checkpoint[k])
        v.eval()
    
    driving_paths = glob.glob(os.path.join(args.natural_d, '*'))

    print(driving_paths)
    
    vs = Visualizer()

    for idx, dri_path in enumerate(driving_paths):
        os.makedirs(args.natural_o + '/' + os.path.basename(dri_path), exist_ok=True)
        img_paths = glob.glob(dri_path + '/' + '*.png') 
        # s = img_as_float32(io.imread(img_paths[0]))[:, :, :3]
        # s = np.array(s, dtype="float32").transpose((2, 0, 1))
        # s = torch.from_numpy(s).cuda().unsqueeze(0)
        # s = F.interpolate(s, size=(256, 256))
        # fs = g_models["afe"](s)
        # kp_c = g_models["ckd"](s)
        # _, _, _, t, scale = g_models["hpe_ede"](s)
        # with torch.no_grad():
        #     hp.eval()
        #     yaw, pitch, roll = hp(F.interpolate(apply_imagenet_normalization(s), size=(224, 224)))
        # # kp_s, Rs = transform_kp(kp_c, yaw, pitch, roll, t, scale)
        # # kp_s, _, _, _, _ = g_models["efe"](s, None, kp_s)
        # delta_s, _, _, _, _ = g_models["efe"](s, None, kp_c)
        # kp_s, Rs = transform_kp(kp_c+delta_s, yaw, pitch, roll, t, scale)
        
        for ind, dri in tqdm.tqdm(enumerate(img_paths), total=len(img_paths)):
            save_path = args.natural_o + '/' + os.path.basename(dri_path) + '/' + os.path.basename(dri)
            img = img_as_float32(io.imread(dri))[:, :, :3]
            img = np.array(img, dtype="float32").transpose((2, 0, 1))
            img = torch.from_numpy(img).cuda().unsqueeze(0)
            _, _, _, t, scale = g_models["hpe_ede"](img)
            fimg = g_models["afe"](img)
            with torch.no_grad():
                hp.eval()
                yaw, pitch, roll = hp(F.interpolate(apply_imagenet_normalization(img), size=(224, 224)))
            
            kp_c_d = g_models["ckd"](img)
            # delta = delta
            delta_d, _, _, _, _ = g_models["efe"](img, None, kp_c_d)
            kp_d1 = kp_c_d
            # kp_d2 = kp_c + delta_d
            kp_d3, Rd3 = transform_kp(kp_c_d + delta_d, yaw*0, pitch*0, roll*0, t*0, scale*0.85)

            kp_d6, Rd6 = transform_kp(kp_c_d + delta_d, yaw, pitch, roll, t, scale)


            # deformation, occlusion, mask = g_models["mfe"](fs, kp_s, kp_d8, Rs, Rd)
            # generated_d = g_models["generator"](fs, deformation, occlusion)
            deformation, occlusion, _ = g_models["mfe"](fimg, kp_d6, kp_d1, Rd6, Rd3)
            generated_d1, _, _ = g_models["generator"](fimg, deformation, occlusion)
            # deformation, occlusion, _ = g_models["mfe"](fs, kp_s, kp_d2, Rs, Rd3)
            # generated_d2, _, _ = g_models["generator"](fs, deformation, occlusion)
            # deformation, occlusion, _ = g_models["mfe"](fs, kp_s, kp_d3, Rs, Rd3)
            # generated_d3, _, _ = g_models["generator"](fs, deformation, occlusion)	
            # deformation, occlusion, _ = g_models["mfe"](fs, kp_s, kp_d4, Rs, Rd4)
            # generated_d4, _, _ = g_models["generator"](fs, deformation, occlusion)
            # deformation, occlusion, _ = g_models["mfe"](fs, kp_s, kp_d5, Rs, Rd5)
            # generated_d5, _, _ = g_models["generator"](fs, deformation, occlusion)
            # deformation, occlusion, _ = g_models["mfe"](fs, kp_s, kp_d6, Rs, Rd6)
            # generated_d6, _, _ = g_models["generator"](fs, deformation, occlusion)
            # deformation, occlusion, _ = g_models["mfe"](fs, kp_s, kp_d7, Rs, Rd7)
            # generated_d7, _, _ = g_models["generator"](fs, deformation, occlusion)
            # deformation, occlusion, _ = g_models["mfe"](fs, kp_s, kp_d8, Rs, Rd8)
            # generated_d8, _, _ = g_models["generator"](fs, deformation, occlusion)

            img_np = img.data.cpu()
            kp_img_np = kp_d6.data.cpu().numpy()[:, :, :2]
            img_np = np.transpose(img_np, [0, 2, 3, 1])

            kp_d1_np = kp_d1.data.cpu().numpy()[:, :, :2]
            
            generated_d1_np = generated_d1.data.cpu().numpy().transpose([0, 2, 3, 1])
            
            img_d = [(img_np, kp_img_np),
                    (generated_d1_np, kp_d1_np)]

            img_d = vs.create_image_grid(*img_d)
            img_d = (255 * img_d).astype(np.uint8)
            cv2.imwrite(save_path, img_d[:,:,::-1])

def edit_one_img(s_path, d_path):
    vs = Visualizer()
    
    hp = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66).cuda()
    hp.load_state_dict(torch.load("hopenet_robust_alpha1.pkl", map_location=torch.device("cpu")))
    for parameter in hp.parameters():
        parameter.requires_grad = False
    g_models = {"efe": EFE(), "afe": AFE(), "ckd": CKD(), "hpe_ede": HPE_EDE(), "mfe": MFE(), "generator": Generator()}
    ckp_path = os.path.join(args.ckp_dir, "%s-checkpoint.pth.tar" % str(args.ckp).zfill(8))
    checkpoint = torch.load(ckp_path, map_location=torch.device("cpu"))
    for k, v in g_models.items():
        v.cuda()
        v.load_state_dict(checkpoint[k])
        v.eval()
        
    s = img_as_float32(io.imread(s_path))[:, :, :3]
    s = np.array(s, dtype="float32").transpose((2, 0, 1))
    s = torch.from_numpy(s).cuda().unsqueeze(0)
    s = F.interpolate(s, size=(256, 256))
    
    fs = g_models["afe"](s)
    kp_c = g_models["ckd"](s)
    _, _, _, t_s, scale_s = g_models["hpe_ede"](s)
    with torch.no_grad():
        hp.eval()
        yaw_s, pitch_s, roll_s = hp(F.interpolate(apply_imagenet_normalization(s), size=(224, 224)))
    # kp_s, Rs = transform_kp(kp_c, yaw, pitch, roll, t, scale)
    # kp_s, _, _, _, _ = g_models["efe"](s, None, kp_s)
    delta_s, _, _, _, _ = g_models["efe"](s, None, kp_c)
    kp_s, Rs = transform_kp(kp_c+delta_s, yaw_s, pitch_s, roll_s, t_s, scale_s)
    
    img = img_as_float32(io.imread(d_path))[:, :, :3]
    img = np.array(img, dtype="float32").transpose((2, 0, 1))
    img = torch.from_numpy(img).cuda().unsqueeze(0)
    img = F.interpolate(img, size=(256, 256))
    fimg = g_models["afe"](img)
    _, _, _, t_d, scale_d = g_models["hpe_ede"](img)
    with torch.no_grad():
        hp.eval()
        yaw_d, pitch_d, roll_d = hp(F.interpolate(apply_imagenet_normalization(img), size=(224, 224)))
    
    kp_c_d = g_models["ckd"](img)
    kp_c_d = kp_c
    # delta = delta
    delta_d, _, _, _, _ = g_models["efe"](img, None, kp_c_d)
    kp_d,  Rd  = transform_kp(kp_c_d+delta_d, yaw_d, pitch_d, roll_d, t_d, scale_d)

    kp_d0, Rd0 = transform_kp(kp_c_d, yaw_d*0., pitch_d*0., roll_d*0., t_d*0., scale_d)
    kp_d1, Rd1 = transform_kp(kp_c_d, yaw_d*0., pitch_d*0., roll_d*0., t_d, scale_d)
    kp_d2, Rd2 = transform_kp(kp_c_d, yaw_d, pitch_d, roll_d, t_d*0., scale_d)
    kp_d3, Rd3 = transform_kp(kp_c_d+delta_d, yaw_d*0., pitch_d*0., roll_d*0., t_d*0., scale_d)
    kp_d4, Rd4 = transform_kp(kp_c_d+delta_d, yaw_d*0., pitch_d*0., roll_d*0., t_d, scale_d)
    kp_d5, Rd5 = transform_kp(kp_c_d, yaw_d, pitch_d, roll_d, t_d, scale_d)
    kp_d6, Rd6 = transform_kp(kp_c_d+delta_d, yaw_d, pitch_d, roll_d, t_d*0, scale_d)
    kp_d7, Rd7 = transform_kp(kp_c_d+delta_d, yaw_d, pitch_d, roll_d, t_d, scale_d)
    
    
    deformation0, occlusion0, _ = g_models["mfe"](fs, kp_s, kp_d0, Rd, Rd0) # exp
    generated_d0, _, _ = g_models["generator"](fs, deformation0, occlusion0)
    
    deformation1, occlusion1, _ = g_models["mfe"](fs, kp_s, kp_d1, Rs, Rd1) # pose
    generated_d1, _, _ = g_models["generator"](fs, deformation1, occlusion1)
    
    deformation2, occlusion2, _ = g_models["mfe"](fs, kp_s, kp_d2, Rs, Rd2) # pose + trans
    generated_d2, _, _ = g_models["generator"](fs, deformation2, occlusion2)
    
    deformation3, occlusion3, _ = g_models["mfe"](fs, kp_s, kp_d3, Rs, Rd3) # pose + trans + scale
    generated_d3, _, _ = g_models["generator"](fs, deformation3, occlusion3) 
        
    deformation4, occlusion4, _ = g_models["mfe"](fs, kp_s, kp_d4, Rs, Rd4) # pose + trans + scale
    generated_d4, _, _ = g_models["generator"](fs, deformation4, occlusion4) 
    
    deformation5, occlusion5, _ = g_models["mfe"](fs, kp_s, kp_d5, Rs, Rd5) # pose + trans + scale
    generated_d5, _, _ = g_models["generator"](fs, deformation5, occlusion5) 
    
    deformation6, occlusion6, _ = g_models["mfe"](fs, kp_s, kp_d6, Rs, Rd6) # pose + trans + scale
    generated_d6, _, _ = g_models["generator"](fs, deformation6, occlusion6) 
    
    deformation7, occlusion7, _ = g_models["mfe"](fs, kp_s, kp_d7, Rs, Rd7) # pose + trans + scale
    generated_d7, _, _ = g_models["generator"](fs, deformation7, occlusion7) 
    

    
    s_np = s.data.cpu()
    s_np = np.transpose(s_np, [0, 2, 3, 1])

    img_np = img.data.cpu()
    img_np = np.transpose(img_np, [0, 2, 3, 1])

    generated_d0_np = generated_d0.data.cpu().numpy().transpose([0, 2, 3, 1])
    generated_d1_np = generated_d1.data.cpu().numpy().transpose([0, 2, 3, 1])
    generated_d2_np = generated_d2.data.cpu().numpy().transpose([0, 2, 3, 1])
    generated_d3_np = generated_d3.data.cpu().numpy().transpose([0, 2, 3, 1])
    generated_d4_np = generated_d4.data.cpu().numpy().transpose([0, 2, 3, 1])
    generated_d5_np = generated_d5.data.cpu().numpy().transpose([0, 2, 3, 1])
    generated_d6_np = generated_d6.data.cpu().numpy().transpose([0, 2, 3, 1])
    generated_d7_np = generated_d7.data.cpu().numpy().transpose([0, 2, 3, 1])
    
    
    img_d = [s_np, img_np, 
             generated_d0_np, generated_d1_np, 
             generated_d2_np, generated_d3_np,
             generated_d4_np, generated_d5_np,
             generated_d6_np, generated_d7_np]
    
    img_d = vs.create_image_grid(*img_d)
    img_d = (255 * img_d).astype(np.uint8)
    imageio.imwrite('demo/2/one_motivation_test/test145.png', img_d)


def edit_one_img_kp(s_path, d_path):
    vs = Visualizer()
    
    hp = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66).cuda()
    hp.load_state_dict(torch.load("hopenet_robust_alpha1.pkl", map_location=torch.device("cpu")))
    for parameter in hp.parameters():
        parameter.requires_grad = False
    g_models = {"efe": EFE(), "afe": AFE(), "ckd": CKD(), "hpe_ede": HPE_EDE(), "mfe": MFE(), "generator": Generator()}
    ckp_path = os.path.join(args.ckp_dir, "%s-checkpoint.pth.tar" % str(args.ckp).zfill(8))
    checkpoint = torch.load(ckp_path, map_location=torch.device("cpu"))
    for k, v in g_models.items():
        v.cuda()
        v.load_state_dict(checkpoint[k])
        v.eval()
        
    s = img_as_float32(io.imread(s_path))[:, :, :3]
    s = np.array(s, dtype="float32").transpose((2, 0, 1))
    s = torch.from_numpy(s).cuda().unsqueeze(0)
    s = F.interpolate(s, size=(256, 256))
    
    fs = g_models["afe"](s)
    kp_c = g_models["ckd"](s)
    _, _, _, t_s, scale_s = g_models["hpe_ede"](s)
    with torch.no_grad():
        hp.eval()
        yaw_s, pitch_s, roll_s = hp(F.interpolate(apply_imagenet_normalization(s), size=(224, 224)))
    # kp_s, Rs = transform_kp(kp_c, yaw, pitch, roll, t, scale)
    # kp_s, _, _, _, _ = g_models["efe"](s, None, kp_s)
    delta_s, _, _, _, _ = g_models["efe"](s, None, kp_c)
    kp_s, Rs = transform_kp(kp_c+delta_s, yaw_s, pitch_s, roll_s, t_s, scale_s)
    
    img = img_as_float32(io.imread(d_path))[:, :, :3]
    img = np.array(img, dtype="float32").transpose((2, 0, 1))
    img = torch.from_numpy(img).cuda().unsqueeze(0)
    img = F.interpolate(img, size=(256, 256))
    fimg = g_models["afe"](img)
    _, _, _, t_d, scale_d = g_models["hpe_ede"](img)
    with torch.no_grad():
        hp.eval()
        yaw_d, pitch_d, roll_d = hp(F.interpolate(apply_imagenet_normalization(img), size=(224, 224)))
    
    kp_c_d = g_models["ckd"](img)
    kp_c_d = kp_c
    # delta = delta
    delta_d, _, _, _, _ = g_models["efe"](img, None, kp_c_d)
    kp_d,  Rd  = transform_kp(kp_c_d+delta_d, yaw_d, pitch_d, roll_d, t_d, scale_d)

    kp_d0, Rd0 = transform_kp(kp_c_d, yaw_d*0., pitch_d*0., roll_d*0., t_d*0., scale=1) # neutral
    kp_d1, Rd1 = transform_kp(kp_c_d+delta_d, yaw_d*0., pitch_d*0., roll_d*0., t_d*0., scale=1) # neutral + delta
    kp_d2, Rd2 = transform_kp(kp_c_d+delta_d, yaw_d*0., pitch_d*0., roll_d*0., t_d*0., scale_d) # neutral + delta + scale
    kp_d3, Rd3 = transform_kp(kp_c_d+delta_d, yaw_d, pitch_d, roll_d, t_d*0., scale_d) # neutral + delta + scale + rotation
    kp_d4, Rd4 = transform_kp(kp_c_d+delta_d, yaw_d, pitch_d, roll_d, t_d, scale_d) # neutral + delta + scale + rotation + translation
    
    kp_d5, Rd5 = transform_kp(kp_c_d, yaw_d, pitch_d, roll_d, t_d, scale_d) # scale + translation + rotation
    kp_d6, Rd6 = transform_kp(kp_c_d+delta_d, yaw_d, pitch_d, roll_d, t_d*0, scale_d) # scale + rotation + delta
    kp_d7, Rd7 = transform_kp(kp_c_d+delta_d, yaw_d, pitch_d, roll_d, t_d, scale_d) # scale + translation + rotation + delta
    kp_d8, Rd8 = transform_kp(kp_c_d, yaw_d*0., pitch_d*0., roll_d*0., t_d*0., scale=1) # neutral

    
    
    deformation0, occlusion0, _ = g_models["mfe"](fs, kp_s, kp_d0, Rd, Rd0) # exp
    generated_d0, _, _ = g_models["generator"](fs, deformation0, occlusion0)
    
    deformation1, occlusion1, _ = g_models["mfe"](fs, kp_s, kp_d1, Rs, Rd1) # pose
    generated_d1, _, _ = g_models["generator"](fs, deformation1, occlusion1)
    
    deformation2, occlusion2, _ = g_models["mfe"](fs, kp_s, kp_d2, Rs, Rd2) # pose + trans
    generated_d2, _, _ = g_models["generator"](fs, deformation2, occlusion2)
    
    deformation3, occlusion3, _ = g_models["mfe"](fs, kp_s, kp_d3, Rs, Rd3) # pose + trans + scale
    generated_d3, _, _ = g_models["generator"](fs, deformation3, occlusion3) 
        
    deformation4, occlusion4, _ = g_models["mfe"](fs, kp_s, kp_d4, Rs, Rd4) # pose + trans + scale
    generated_d4, _, _ = g_models["generator"](fs, deformation4, occlusion4) 
    
    deformation5, occlusion5, _ = g_models["mfe"](fs, kp_s, kp_d5, Rs, Rd5) # pose + trans + scale
    generated_d5, _, _ = g_models["generator"](fs, deformation5, occlusion5) 
    
    deformation6, occlusion6, _ = g_models["mfe"](fs, kp_s, kp_d6, Rs, Rd6) # pose + trans + scale
    generated_d6, _, _ = g_models["generator"](fs, deformation6, occlusion6) 
    
    deformation7, occlusion7, _ = g_models["mfe"](fs, kp_s, kp_d7, Rs, Rd7) # pose + trans + scale
    generated_d7, _, _ = g_models["generator"](fs, deformation7, occlusion7) 
    
    deformation8, occlusion8, _ = g_models["mfe"](fs, kp_s, kp_d8, Rs, Rd8) # pose + trans + scale
    generated_d8, _, _ = g_models["generator"](fs, deformation8, occlusion8) 
    
    s_np = s.data.cpu()
    s_np = np.transpose(s_np, [0, 2, 3, 1])

    img_np = img.data.cpu()
    img_np = np.transpose(img_np, [0, 2, 3, 1])

    kp_s_np = kp_s.data.cpu().numpy()[:, :, :2]
    kp_d_np = kp_d.data.cpu().numpy()[:, :, :2]

    kp_d0_np = kp_d0.data.cpu().numpy()[:, :, :2]
    kp_d1_np = kp_d1.data.cpu().numpy()[:, :, :2]
    kp_d2_np = kp_d2.data.cpu().numpy()[:, :, :2]
    kp_d3_np = kp_d3.data.cpu().numpy()[:, :, :2]
    kp_d4_np = kp_d4.data.cpu().numpy()[:, :, :2]
    kp_d5_np = kp_d5.data.cpu().numpy()[:, :, :2]
    kp_d6_np = kp_d6.data.cpu().numpy()[:, :, :2]
    kp_d7_np = kp_d7.data.cpu().numpy()[:, :, :2]
    kp_d8_np = kp_d8.data.cpu().numpy()[:, :, :2]

    generated_d0_np = generated_d0.data.cpu().numpy().transpose([0, 2, 3, 1])
    generated_d1_np = generated_d1.data.cpu().numpy().transpose([0, 2, 3, 1])
    generated_d2_np = generated_d2.data.cpu().numpy().transpose([0, 2, 3, 1])
    generated_d3_np = generated_d3.data.cpu().numpy().transpose([0, 2, 3, 1])
    generated_d4_np = generated_d4.data.cpu().numpy().transpose([0, 2, 3, 1])
    generated_d5_np = generated_d5.data.cpu().numpy().transpose([0, 2, 3, 1])
    generated_d6_np = generated_d6.data.cpu().numpy().transpose([0, 2, 3, 1])
    generated_d7_np = generated_d7.data.cpu().numpy().transpose([0, 2, 3, 1])
    generated_d8_np = generated_d8.data.cpu().numpy().transpose([0, 2, 3, 1])
    
    
    img_d = [(s_np, kp_s_np), (img_np, kp_d_np), (generated_d0_np, kp_d0_np),
        (generated_d1_np, kp_d1_np), (generated_d2_np, kp_d2_np), (generated_d3_np, kp_d3_np), 
        (generated_d4_np, kp_d4_np), (generated_d5_np, kp_d5_np), (generated_d6_np, kp_d6_np),
        (generated_d7_np, kp_d7_np), (generated_d8_np, kp_d8_np), generated_d7_np]
    
    img_d = vs.create_image_grid(*img_d)
    img_d = (255 * img_d).astype(np.uint8)
    imageio.imwrite('demo/2/one_motivation_test/test252.png', img_d)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser(description="face-vid2vid")

    def str2bool(s):
        return s.lower().startswith("t")

    # parser.add_argument("--ckp_dir", type=str, default="ckp_1644_mainv9finalv3-lml-dls-newgenmodel-mask", help="Checkpoint dir")
    parser.add_argument("--ckp_dir", type=str, default="ckp_1644_mainv9finalv3-lml-dls-newgenmodel-mask", help="Checkpoint dir")
    parser.add_argument("--output", type=str, default="output.gif", help="Output video")
    parser.add_argument("--ckp", type=int, default=145, help="Checkpoint epoch")
    parser.add_argument("--source", type=str, default="demo/s", help="Source image, f for face frontalization, r for reconstruction")
    parser.add_argument("--driving", type=str, default='demo/out/rec/newgen-mask-lw_181/ENGLAND_00035.mp4', help="Driving dir")
    parser.add_argument("--num_frames", type=int, default=90, help="Number of frames")
    parser.add_argument("--natural_d", type=str, default='demo/2/d-png', help="Driving dir")
    parser.add_argument("--natural_o", type=str, default='demo/2/n-png-239', help="Driving dir")

    parser.add_argument("--test_d", type=str, default='demo/2/d-png/id10468#EICQVj70ePE#00007.txt#000.mp4/00000020.png', help="Driving dir")
    # parser.add_argument("--test_d", type=str, default='demo/2/d-png/id10751#U_kqhmtzZHU#00001.txt#000.mp4/00000276.png', help="Driving dir")
    # 2
    # parser.add_argument("--test_s", type=str, default='/data1/code/lh/1.faceanimation/One-Shot_Free-View_Neural_Talking_Head_Synthesis/demo/2/d-png/id10406#VzA66aG_-Lw#00013.txt#000.mp4/00000026.png', help="Driving dir")
    # parser.add_argument("--test_s", type=str, default='/data1/code/lh/1.faceanimation/One-Shot_Free-View_Neural_Talking_Head_Synthesis/demo/2/d-png/id10406#VzA66aG_-Lw#00013.txt#000.mp4/00000026.png', help="Driving dir")

    # 1 
    # parser.add_argument("--test_d", type=str, default='/data1/datasets/vox1/vox-png/train/id10038#d-SooYXE3Fc#00004.txt#000.mp4/0000138.png', help="Driving dir")
    # parser.add_argument("--test_d", type=str, default='/data1/datasets/vox1/vox-png/train/id10038#d-SooYXE3Fc#00009.txt#000.mp4/0000133.png', help="Driving dir")


    # 1
    # parser.add_argument("--test_s", type=str, default='demo/2/d-png/id10468#EICQVj70ePE#00007.txt#000.mp4/00000223.png', help="Driving dir")
    # parser.add_argument("--test_s", type=str, default='/data1/datasets/vox1/vox-png/train/id10468#EICQVj70ePE#00005.txt#000.mp4/0000119.png', help="Driving dir")

    parser.add_argument("--test_s", type=str, default='/data1/datasets/vox1/vox-png/train/id10071#A2bWI0qrkd8#00005.txt#000.mp4/0000103.png', help="Driving dir")
    # parser.add_argument("--test_d", type=str, default='/data1/code/lh/1.faceanimation/One-Shot_Free-View_Neural_Talking_Head_Synthesis/demo/2/d-png/id10468#EICQVj70ePE#00007.txt#000.mp4/00000126.png', help="Driving dir")
    # parser.add_argument("--test_s", type=str, default='/data/repo/code/lh/2.faceanimation/face-vae/demo/2/d-png/id11248#yDqlBD8m_b8#00001.txt#000.mp4/00000051.png', help="Driving dir")
    # parser.add_argument("--test_s", type=str, default='/data/repo/code/lh/2.faceanimation/face-vae/demo/2/d-png/id11248#yDqlBD8m_b8#00001.txt#000.mp4/00000002.png', help="Driving dir")


    args = parser.parse_args()
    # eval(args)
    # demo(args)
    # natural(args)
    edit_one_img(args.test_s, args.test_d)
    # edit_one_img_kp(args.test_s, args.test_d)

    
    
