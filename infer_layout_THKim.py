import os, sys, time, glob
import argparse
import importlib
from tqdm import tqdm
from imageio import imread, imwrite
import torch
import numpy as np
import matplotlib.pyplot as plt 
from lib.config import config, update_config
import cv2

if __name__ == '__main__':

    # Parse args & config
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', default="config/mp3d_layout/HOHO_layout_aug_efficienthc_Transen1_resnet34.yaml")
    parser.add_argument('--pth', default="ckpt/mp3d_layout_HOHO_layout_aug_efficienthc_Transen1_resnet34/ep300.pth")
    parser.add_argument('--out', default="assets/")
    # parser.add_argument('--inp', default="assets/pano_asmasuxybohhcj.png")
    parser.add_argument('--inp', default="assets/1.png")

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    
    
    # @@@ THKIM @@@
    args.inp = "assets/16-47-04.jpg"
    
    
    update_config(config, args)
    device = 'cuda' if config.cuda else 'cpu'
    print(torch.cuda.is_available())

    # Parse input paths
    rgb_lst = glob.glob(args.inp)
    if len(rgb_lst) == 0:
        print('No images found')
        import sys; sys.exit()

    # Init model
    model_file = importlib.import_module(config.model.file)
    model_class = getattr(model_file, config.model.modelclass)
    net = model_class(**config.model.kwargs)
    net.load_state_dict(torch.load(args.pth, map_location=device))
    net = net.eval().to(device)

    # Run inference
    with torch.no_grad():
        for path in tqdm(rgb_lst):
            # rgb = imread(path)
            
            rgb = cv2.imread(path)
            rgb = cv2.resize(rgb, (1024, 512), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            x = torch.from_numpy(rgb).permute(2,0,1)[None].float() / 255.
            x = x.to(device)
            output = net.infer(x)
            cor_id = output['cor_id']
            y_bon_ = output['y_bon_']
            y_cor_ = output['y_cor_']

            fname = os.path.splitext(os.path.split(path)[1])[0]
            
            # save .txt
            with open(os.path.join(args.out, f'{fname}.layout.txt'), 'w') as f:
                for u, v in cor_id:
                    f.write(f'{u:.1f} {v:.1f}\n')
            with open(os.path.join(args.out, f'{fname}.y_bon_.txt'), 'w') as f:
                for i in range(len(y_bon_[0])):
                    f.write(f'{y_bon_[0, i]:.1f} {y_bon_[1, i]:.1f}\n')
            with open(os.path.join(args.out, f'{fname}.y_cor_.txt'), 'w') as f:
                for u in y_cor_:
                    f.write(f'{u:.1f}\n')
                    
            # save .png
            plt.figure(figsize=(24,10))
            plt.subplot(221)
            plt.imshow(np.concatenate([
                (y_cor_ * 255).reshape(1,-1,1).repeat(30, 0).repeat(3, 2).astype(np.uint8), rgb[30:]], axis=0))
            plt.plot(np.arange(y_bon_.shape[1]), y_bon_[0], 'r-')
            plt.plot(np.arange(y_bon_.shape[1]), y_bon_[1], 'r-')
            plt.scatter(cor_id[:, 0], cor_id[:, 1], marker='x', c='b')
            plt.axis('off')
            plt.title('y_bon_ (red) / y_cor_ (up-most bar) / cor_id (blue x)')
            plt.savefig(os.path.join(args.out, f'{fname}.img.png'), bbox_inches='tight')
            plt.show()
