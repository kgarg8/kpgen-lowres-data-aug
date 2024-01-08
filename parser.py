import argparse
from argparse import ArgumentParser

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = ArgumentParser(description="Seq2Seq")
    parser.add_argument('--model', type=str, default='GRUSeq2Seq', choices=['GRUSeq2Seq', 'LEDSeq2Seq', 'BartSeq2Seq'])
    parser.add_argument('--model_type', type=str, default='Seq2Seq', choices=['Seq2Seq'])
    parser.add_argument('--dataset', type=str, default='kp20k', choices=['LDKP10km-lr-1000', 'LDKP10km-lr-2000', 'LDKP10km-lr-4000', 'LDKP10km-lr-8000', 'KPTimes-lr-1000', 'KPTimes-lr-2000', 'KPTimes-lr-4000', 'KPTimes-lr-8000', 'LDKP3km-lr-1000', 'LDKP3km-lr-2000', 'LDKP3km-lr-4000', 'LDKP3km-lr-8000']) 
    parser.add_argument('--lr_seed', type=int, default=1433, help='hint: [1433, 1896, 1922]')
    parser.add_argument('--version', type=str, default='v1', choices=['v2', 'v6', 'cat_v6', 'v2_kpdrop_aug', 'v6_kpdrop_aug', 'v2_bt_aug', 'v2_sr_aug_k1', 'v2_sr_aug_k2', 'v2_sr_aug_k-1', 'v6_sr_aug_k-1', 'v6_kpdrop_aug_P1', 'v2_bt_drop_aug', 'v2_std_sr_aug', 'v6_std_sr_aug', 'v6_bt_aug'])
    parser.add_argument('--test', type=str2bool, default=False, const=True, nargs='?')
    parser.add_argument('--times', type=int, default=1)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--initial_time', type=int, default=0)
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--no_display', type=str2bool, default=False, const=True, nargs='?')
    parser.add_argument('--display_params', type=str2bool, default=True, const=True, nargs='?')
    parser.add_argument('--display_step', type=int, default=100)
    parser.add_argument('--example_display_step', type=int, default=500)
    parser.add_argument('--checkpoint', type=str2bool, help="checkpointing is done always whether you want it or not but this option can be used to load existing checkpoint", default=False)
    return parser