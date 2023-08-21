import pathlib

import cv2
import torch
import PIL
from os import path as osp
from easydict import EasyDict as edict

from graphau.OpenGraphAU.conf import get_config
from graphau.OpenGraphAU.dataset import pil_loader
from graphau.OpenGraphAU.utils import hybrid_prediction_infolist, load_state_dict, image_eval
from graphau.utils import HOMEDIR


class GraphAUDetector:
    def __init__(self, device):
        self.device = torch.device(device)
        self.conf = {'dataset': 'hybrid',
                     'batch_size': 64,
                     'learning_rate': 1e-05,
                     'epochs': 20,
                     'num_workers': 4,
                     'weight_decay': 0.0005,
                     'optimizer_eps': 1e-08,
                     'crop_size': 224,
                     'arc': 'swin_transformer_base',
                     'metric': 'dots',
                     'lam': 0.001,
                     'resume': osp.join(HOMEDIR, 'checkpoints/OpenGprahAU-SwinB_second_stage.pth'),
                     'stage': 2,
                     'dataset_path': 'data/BP4D',
                     'num_classes': 41,
                     'num_main_classes': 27,
                     'num_sub_classes': 14,
                     'neighbor_num': 4,
                     'evaluate': True}
        if self.device.type != 'cpu':
            self.conf['gpu_ids'] = str(self.device.index)
        self.aus = ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10', 'AU11', 'AU12', 'AU13', 'AU14', 'AU15', 'AU16', 'AU17',
                    'AU18', 'AU19', 'AU20', 'AU22', 'AU23', 'AU24', 'AU25', 'AU26', 'AU27', 'AU32', 'AU38', 'AU39', 'AUL1', 'AUR1',
                    'AUL2', 'AUR2', 'AUL4', 'AUR4', 'AUL6', 'AUR6', 'AUL10', 'AUR10', 'AUL12', 'AUR12', 'AUL14', 'AUR14']
        self.conf = edict(self.conf)

        if self.conf.stage == 1:
            from graphau.OpenGraphAU.model.ANFL import MEFARG
            self.net = MEFARG(num_main_classes=self.conf.num_main_classes, num_sub_classes=self.conf.num_sub_classes, backbone=self.conf.arc, neighbor_num=self.conf.neighbor_num, metric=self.conf.metric)
        else:
            from graphau.OpenGraphAU.model.MEFL import MEFARG
            self.net = MEFARG(num_main_classes=self.conf.num_main_classes, num_sub_classes=self.conf.num_sub_classes, backbone=self.conf.arc)

        if self.conf.resume != '':
            self.net = load_state_dict(self.net, self.conf.resume)
        self.net = self.net.to(self.device)
        self.net.eval()
        self.img_transform = image_eval()

    def detect(self, frame, return_names=False):
        # img = pil_loader(img_path)
        img = PIL.Image.fromarray(frame)
        img = self.img_transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.net(img)
            pred = pred.squeeze().cpu().numpy()
        if return_names:
            return list(zip(self.aus, pred))
        else:
            return pred


if __name__ == '__main__':
    img_path = r'OpenGraphAU/demo_imgs/1014.jpg'
    detector = GraphAUDetector('cpu')
    pred = detector.detect(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    print(pred)
