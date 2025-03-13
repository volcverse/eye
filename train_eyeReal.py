import datetime
import os
import time
import torch
import torch.utils.data
import math
from model.network import EyeRealNet
from torchvision import transforms as T
import data.funcs as funcs
import cv2
from data.sampler import *
import gc
from data.funcs import is_main_process
import warnings
from config.scene_dict import *
from model.loss import get_aux_loss
warnings.filterwarnings('ignore')

def sort_key(s):
    # "pair0_x5.896_y45.396_z-24.465_left.jpg"
    pairs = s.split('_')
    index = int(pairs[0][4:])
    direction = int(pairs[-1][0])
    return (index, direction)


def get_dataset(image_set, transform, args, idx=None):
   from data.dataset import SceneDataset

   data_prefix = os.path.join(os.getcwd(), args.data_path)
   images_path = os.listdir(args.data_path)
   images_path = sorted(images_path, key=sort_key)
   # images_label = args.data_path+'.json'

   ds = SceneDataset(images_path=images_path,
                  data_prefix=data_prefix,
                  transform=transform,
                  delta=[args.delta_x, args.delta_y, args.delta_z],
                  vertical=args.vertical)
   return ds

def get_transform(args):
    transforms = [
        T.Resize((args.image_height, args.image_width)),
        T.ToTensor(),
        # T.Normalize(
        #     mean=[0.485, 0.456, 0.406], 
        #     std=[0.229, 0.224, 0.225]),
    ]

    return T.Compose(transforms)

def get_grad_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def save_from_preds(output, ckpt_path):
    save_path_layer = ckpt_path+'/preditions/layers/'
    save_path_view = ckpt_path+'/preditions/views/'
    os.makedirs(save_path_layer, exist_ok=True)
    os.makedirs(save_path_view, exist_ok=True)

    patterns, preds = output
    
    for i, pred in enumerate(patterns):
        pred = pred.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).type(torch.uint8).cpu().numpy()
        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path_layer+'layer-'+str(i+1)+'.png', pred)
    
    for i, pred in enumerate(preds):
        pred = pred.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).type(torch.uint8).cpu().numpy()
        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path_view+'view-'+str(i+1)+'.png', pred)

def save_from_preds_in_wandb(output, caption=''):
    patterns, preds, gts, baseline = output
    
    for i, pred in enumerate(patterns):
        pred = pred.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).type(torch.uint8).cpu().numpy()
        img_wandb = wandb.Image(pred, caption=caption)
        wandb.log({'layer-'+str(i+1): img_wandb})
    
    for i, pred in enumerate(preds):
        pred = pred.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).type(torch.uint8).cpu().numpy()
        img_wandb = wandb.Image(pred, caption=caption)
        wandb.log({'view-'+str(i+1): img_wandb})

    for i, pred in enumerate(gts):
        pred = pred.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).type(torch.uint8).cpu().numpy()
        img_wandb = wandb.Image(pred, caption=caption)
        wandb.log({'view_gt-'+str(i+1): img_wandb})
    
def train_one_epoch(args, model: EyeRealNet, optimizer, data_loader, lr_scheduler, 
                    epoch, print_freq, iterations, save_preds=False, ckpt_path='./'):
    model.train()
    metric_logger = funcs.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', funcs.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    train_PSNR = 0
    total_its = 0

    for data in metric_logger.log_every(data_loader, print_freq, header):
        # images : 2 k 3 w h
        images, views = data
        images, views = images.cuda(non_blocking=True), views.cuda(non_blocking=True)

        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+


        patterns = model(images, views)
        outs = model.module.get_loss(patterns, gt=images, views=views, return_preds=save_preds) 
        loss = loss_mse = outs['loss_mse']
        PSNR = outs['PSNR']
        if args.mutex:
            loss = outs['loss_mutex']
        elif args.l1_mutex:
            # loss = 0.2*outs['loss_mutex'] + 1*outs['loss_l1']
            loss = args.l1_mutex_ratio*outs['loss_mutex'] + (1-args.l1_mutex_ratio)*outs['loss_l1']
        if args.aux_loss and epoch < int(args.epochs * args.aux_ratio):
            aux_loss = get_aux_loss(patterns, epoch/int(args.epochs * args.aux_ratio), args.aux_weight)
            loss += aux_loss
        loss.backward()

        grad_norm = get_grad_norm(model.module)
        if epoch > 0:
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1, norm_type=2)
        # grad_norm_clip = get_grad_norm(model.module)
        # for name, param in model.module.named_parameters():
        #     if param.grad is None:
        #         print(name)
        optimizer.step()
        
        lr_scheduler.step()

        torch.cuda.synchronize()
        train_loss += loss.item()
        train_PSNR += PSNR
        iterations += 1
        metric_logger.update(loss=loss.item(), PSNR=PSNR, lr=round(optimizer.param_groups[0]["lr"], 8))
        if args.wandb:
            log_dict = {"loss": loss.item(), "loss_mse": loss_mse.item(), "PSNR": PSNR, 
                        "learning_rate": optimizer.param_groups[0]["lr"], "grad_norm": grad_norm}
            if args.aux_loss and args.N_screen != 1 and epoch < int(args.epochs * args.aux_ratio):
                log_dict['aux_loss'] = aux_loss.item()
            if args.add_ssim or args.mutex or args.l1_mutex:
                log_dict['loss_mutex'] = outs['loss_mutex'].item()
                if args.mutex or args.l1_mutex:
                    del log_dict['loss_mse']
                    if args.l1_mutex:
                        log_dict['loss_l1'] = outs['loss_l1'].item()
            wandb.log(log_dict)

        if save_preds:
            patterns = patterns[0].detach().clone()
            preds = outs['preds'][0].detach().clone()
            baseline = images[0].mean(dim=0).detach().clone()
            if total_its % 20 == 0:
                save_from_preds((patterns, preds), ckpt_path)

        # del image, view, preds, loss, data
        total_its += 1
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if args.wandb:
        save_from_preds_in_wandb((patterns, preds, images[0], baseline), caption="epoch: {}".format(epoch))

    return train_loss / total_its, train_PSNR / total_its


def main(args):
    dataset = get_dataset("train", get_transform(args=args), args=args)
    num_ranks = funcs.get_world_size()
    current_rank = funcs.get_rank()
    train_sampler = DistributedRandomSampler(dataset, num_replicas=num_ranks, rank=current_rank,
                                            random_ratio=args.random_ratio, shuffle=True)
    workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 
                   args.workers if not args.debug else 0])
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=workers, pin_memory=args.pin_mem, drop_last=True)
    print(len(data_loader), len(dataset))
    assert len(data_loader) > 0, 'len(dataset) must >= batch_size * gpus'
    cam_distance = 24
    FOV = args.FOV
    if FOV > math.pi:
        FOV = FOV / 180 * math.pi
    model = EyeRealNet(args=args, FOV=FOV)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])#, find_unused_parameters=True)
    single_model = model.module

    if args.ckpt_weights:
        checkpoint = torch.load(args.ckpt_weights, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'])
        args.aux_loss = False
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'])
    
    params_to_optimize = [
        {"params": [p for p in single_model.parameters() if p.requires_grad]},
    ]
    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad)
    # print(args.epochs)
    # print(len(data_loader))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        #  lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)
        lambda x: ((1 + math.cos(x * math.pi / (args.epochs*len(data_loader)))) / 2) * (1 - 0.01) + 0.01)

    start_time = time.time()
    iterations = 0
    best_PSNR = -999
    best_epoch = -1
    exp_name = args.exp_name + '-lr' + str(args.lr) + '-ep' + str(args.epochs) + '-' + time.strftime("%Y-%m-%d-%H:%M:%S")
    ckpt_path = args.output_dir + exp_name
    os.makedirs(ckpt_path, exist_ok=True)

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
    else:
        resume_epoch = -999



    # training loops
    for epoch in range(max(0, resume_epoch+1), args.epochs):
        data_loader.sampler.set_epoch(epoch)

        loss, PSNR = train_one_epoch(args, model, optimizer, data_loader, lr_scheduler, epoch, args.print_freq, iterations, 
                            save_preds=args.save_preds, ckpt_path=ckpt_path)
        PSNR = round(PSNR, 3)
        print('PSNR {}'.format(PSNR))
        save_checkpoint = (best_PSNR < PSNR)
        if save_checkpoint:
            print('Better epoch: {} with PSNR {} than the last best {}\n'.format(epoch, PSNR, best_PSNR))
            dict_to_save = {'model': single_model.state_dict(),
                            'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                            'lr_scheduler': lr_scheduler.state_dict()}

            funcs.save_on_master(dict_to_save, os.path.join(ckpt_path,
                                                            'model_best_{}.pth'.format(args.model_id)))
            best_PSNR = PSNR
            best_epoch = epoch

    if is_main_process():
        os.rename(os.path.join(ckpt_path, 'model_best_{}.pth'.format(args.model_id)), ckpt_path+"/model-epoch{}-{}dB.pth".format(best_epoch, round(best_PSNR, 2)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))




def init_scene_args(args):

    if args.scene in scene_dict:
        arg_dict = scene_dict[args.scene]

        args.scale_physical2world = arg_dict["scale_physical2world"]
        args.thickness = arg_dict["thickness"]
        args.vertical = arg_dict["vertical"]
        args.orientation = arg_dict["orientation"]

        if "physical_width" in arg_dict:
            args.physical_width = arg_dict["physical_width"]
        if "ground_coefficient" in arg_dict:
            args.ground_coefficient = arg_dict["ground_coefficient"]
        if "ground" in arg_dict:
            args.ground = arg_dict["ground"]
        if "delta_x" in arg_dict:
            args.delta_x = arg_dict["delta_x"]
        if "delta_y" in arg_dict:
            args.delta_y = arg_dict["delta_y"]
        if "delta_z" in arg_dict:
            args.delta_z = arg_dict["delta_z"]
    else:
        raise ValueError("wrong input scene name")


if __name__ == "__main__":
    from config.args import get_parser
    parser = get_parser()
    args = parser.parse_args()

    init_scene_args(args=args)


    funcs.init_distributed_mode(args)

    if not is_main_process():
        args.wandb = False
        args.save_preds = False

    if args.wandb:
        import wandb
        wandb.init(project=args.model_id, name=args.exp_name, dir=args.wandb_dir)
    
    main(args)

    if args.wandb:
        wandb.finish()