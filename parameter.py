import argparse
from yacs.config import CfgNode as CN

def get_hparams():
    # configures from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", type=str)
    parser.add_argument("--data", type=str, default='config/blender.yaml')
    parser.add_argument("--test_epoch", type=int, default=1)
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--loss", type=str, nargs="+", default=["Color"])
    parser.add_argument("--Sampling", type=str)
    parser.add_argument("--checkpose", action='store_true', default=False)
    config_cmd = parser.parse_args()

    hparams = CN(new_allowed=True)
    hparams.merge_from_file(f'config/render/{config_cmd.render}.yaml')
    # data config must behind the render config since some parameters maybe overrided
    hparams.merge_from_file(f'config/data/{config_cmd.data}.yaml')
    hparams.test_epoch = config_cmd.test_epoch
    hparams.start_epoch = config_cmd.start_epoch
    # hparams.Loss = config_cmd.loss
    hparams.check_pose = config_cmd.checkpose
    if config_cmd.Sampling:
        hparams.Sampling = config_cmd.Sampling
        

    
    # other default training settings
    hparams.N_workers = 0
    hparams.epoch = 1
    hparams.batch_size = 1024 * 4
    hparams.chunk_size = 1024*32
    hparams.optimizer = 'Adam'
    hparams.scheduler = 'CosineAnnealingLR'
    hparams.node_num = 10

    hparams.save_root = f"save/{hparams.Model}_{hparams.Sampling}_{hparams.Function}_{'_'.join(hparams.Loss)}"
    hparams.ckpt_path = f"{hparams.save_root}/{hparams.data_name}/"
    
    
    # hparams.freeze()
    return(hparams)
    
