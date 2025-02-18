num_epochs = 100
learing_rate = 1e-4
batch_size = 1
num_classes=1 + 46




# 参数设置
initial_temp = 5    # 初始温度
cooling_rate = 0.99     # 冷却速率
min_temp = 1            # 最小温度
initial_solution = 0    # 初始解
beta_rot = 3
beta_pos = 10

theta_scale = 40
pos_scale = 2
DATA_DIR = "/mnt/e/dataset/scenes"
OBJ_DIR = '/mnt/e/dataset/3DFront_p/object/'


import argparse
import torch


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--scene', type=str, default='000ecb5b-b877-4f9a-ab6f-90f385931658')
    parser.add_argument('--dataset', type=str, default='./data/scenes')
    parser.add_argument('--debug',action= "store_true",default=False)
    parser.add_argument('--sketch',type=str,default='./data/sketch/sketch3.jpg')
    parser.add_argument('--epoch',type=int, default=50)
    parser.add_argument('--eps',type=float,default=1e-1)

    parser.add_argument('--sample',type=bool, default=0)
    parser.add_argument('--res', type=int,default=256)

    parser.add_argument('--exp',type=int,default=1)

    parser.add_argument("--settings", type=str, default="scene_1")
    parser.add_argument('--save_interval', type=int, default= 10)
 

    parser.add_argument("--percep_loss", type=str, default="none",
                        help="the type of perceptual loss to be used (L2/LPIPS/none)")
    parser.add_argument("--perceptual_weight", type=float, default=0,
                        help="weight the perceptual loss")
                        
    parser.add_argument("--train_with_clip", type=int, default=0)
    parser.add_argument("--clip_weight", type=float, default=0)
    parser.add_argument("--start_clip", type=int, default=0)
    parser.add_argument("--num_aug_clip", type=int, default=4)
    parser.add_argument("--include_target_in_aug", type=int, default=0)
    parser.add_argument("--augment_both", type=int, default=1,
                        help="if you want to apply the affine augmentation to both the sketch and image")
    parser.add_argument("--vae_loss",type=int, default=0)
    parser.add_argument("--augemntations", type=str, default="affine",
                        help="can be any combination of: 'affine_noise_eraserchunks_eraser_press'")
    parser.add_argument("--noise_thresh", type=float, default=0.5)
    parser.add_argument("--aug_scale_min", type=float, default=0.7)
    parser.add_argument("--force_sparse", type=float, default=0,
                        help="if True, use L1 regularization on stroke's opacity to encourage small number of strokes")
    parser.add_argument("--clip_conv_loss", action= "store_true", default=False)
    parser.add_argument("--clip_conv_loss_type", type=str, default="L2")
    parser.add_argument("--clip_conv_layer_weights",
                        type=str, default="0,0,1.0,1.0,0")
    parser.add_argument("--clip_model_name", type=str, default="RN101")
    parser.add_argument("--clip_fc_loss_weight", type=float, default=1)
    parser.add_argument("--clip_text_guide", type=float, default=0)
    parser.add_argument("--text_target", type=str, default="none")
    parser.add_argument("--orb_loss", action= "store_true", default= False)
    parser.add_argument('--l2_loss', action= "store_true", default=False)
    parser.add_argument('--sample_loss',action= "store_true", default=False)
    parser.add_argument('--semantic_loss',action= "store_true", default=False)
    args = parser.parse_args()

    args.clip_conv_layer_weights = [
        float(item) for item in args.clip_conv_layer_weights.split(',')]
    args.device = torch.device("cuda" if (
            torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    return args

# 类别到颜色的映射表
category_to_color = {
    'Barstool': [1.0, 0.0, 0.0],  # 红色
    'Bookcase / jewelry Armoire': [0.0, 1.0, 0.0],  # 绿色
    'Bunk Bed': [0.0, 0.0, 1.0],  # 蓝色
    'Ceiling Lamp': [1.0, 1.0, 0.0],  # 黄色
    'Chaise Longue Sofa': [1.0, 0.0, 1.0],  # 紫色
    'Children Cabinet': [0.0, 1.0, 1.0],  # 青色
    'Classic Chinese Chair': [1.0, 0.5, 0.0],  # 橙色
    'Coffee Table': [0.5, 0.0, 1.0],  # 紫罗兰色
    'Corner/Side Table': [0.5, 0.5, 0.5],  # 灰色
    'Desk': [1.0, 0.5, 0.5],  # 粉红色
    'Dining Chair': [0.5, 1.0, 0.5],  # 浅绿色
    'Dining Table': [0.5, 0.5, 1.0],  # 浅蓝色
    'Drawer Chest / Corner cabinet': [0.3, 0.3, 0.3],  # 深灰色
    'Dressing Chair': [0.7, 0.7, 0.7],  # 浅灰色
    'Dressing Table': [1.0, 0.75, 0.0],  # 金色
    'Footstool / Sofastool / Bed End Stool / Stool': [0.5, 0.25, 0.0],  # 棕色
    'Kids Bed': [0.2, 0.8, 0.8],  # 水蓝色
    'King-size Bed': [0.8, 0.2, 0.8],  # 品红色
    'L-shaped Sofa': [0.8, 0.8, 0.2],  # 橄榄绿
    'Lazy Sofa': [0.2, 0.2, 0.8],  # 深蓝色
    'Lounge Chair / Cafe Chair / Office Chair': [0.2, 0.8, 0.2],  # 深绿色
    'Loveseat Sofa': [0.8, 0.2, 0.2],  # 深红色
    'Nightstand': [0.2, 0.6, 0.6],  # 蓝绿色
    'Pendant Lamp': [0.6, 0.2, 0.6],  # 紫红色
    'Round End Table': [0.6, 0.6, 0.2],  # 土黄色
    'Shelf': [0.2, 0.4, 0.6],  # 海军蓝
    'Sideboard / Side Cabinet / Console table': [0.4, 0.2, 0.6],  # 靛蓝色
    'Single bed': [0.6, 0.4, 0.2],  # 赭色
    'TV Stand': [0.4, 0.6, 0.2],  # 黄绿色
    'Three-seat / Multi-seat Sofa': [0.2, 0.6, 0.4],  # 蓝绿色
    'Wardrobe': [0.6, 0.2, 0.4],  # 玫瑰红色
    'Wine Cabinet': [0.4, 0.2, 0.4],  # 深紫色
    'armchair': [0.2, 0.4, 0.2],  # 深绿色
}



def semantic_color(obj):
    return category_to_color[obj['coarseSemantic']]

