import torch
import argparse
import numpy as np
import multiprocessing  # Support for multiprocessing

from utils import *
from solver import Solver
from config import get_args, get_config
from data_loader import get_loader

def set_seed(seed):
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        generator = torch.Generator(device='cuda')
        torch.cuda.manual_seed_all(seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        generator = torch.Generator()  # CPU generator

    generator.manual_seed(seed)  # Set the seed for the generator
    return generator  # Return the generator


if __name__ == '__main__':
    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn', force=True)

    # Get parameters
    args = get_args()
# Directory for  data
#Specify the path to your data storage
    dataset = str.lower(args.dataset.strip())
    args.data_dir = '../data'
    pose_dir = '../data'
    audio_dir = '../data'  
    is09_dir = '../data'

    # Set the random seed and get the generator
    generator = set_seed(args.seed)

    print("Start loading the data....")

    # Configure the training, validation, and test data configurations
    train_config = get_config(
        dataset,
        mode='train',
        batch_size=args.batch_size,
        visual_encoder_type=args.visual_encoder_type,
        acoustic_encoder_type=args.acoustic_encoder_type,
        use_transformer_fusion=args.use_transformer_fusion if hasattr(args, 'use_transformer_fusion') else False,
        save_name=args.save_name  # Pass the save_name to the config
    )

    valid_config = get_config(
        dataset,
        mode='valid',
        batch_size=args.batch_size,
        visual_encoder_type=args.visual_encoder_type,
        acoustic_encoder_type=args.acoustic_encoder_type,
        use_transformer_fusion=args.use_transformer_fusion if hasattr(args, 'use_transformer_fusion') else False,
        save_name=args.save_name  # Pass the save_name to the config
    )

    test_config = get_config(
        dataset,
        mode='test',
        batch_size=args.batch_size,
        visual_encoder_type=args.visual_encoder_type,
        acoustic_encoder_type=args.acoustic_encoder_type,
        use_transformer_fusion=args.use_transformer_fusion if hasattr(args, 'use_transformer_fusion') else False,
        save_name=args.save_name  # Pass the save_name to the config
    )

    # Load training, validation, and test data
    train_loader = get_loader(args.data_dir, pose_dir, audio_dir, is09_dir, args.batch_size, phase='train', shuffle=True, generator=generator)
    print('Training data loaded!')

    valid_loader = get_loader(args.data_dir, pose_dir, audio_dir, is09_dir, args.batch_size, phase='valid', shuffle=False, generator=generator)
    print('Validation data loaded!')

    test_loader = get_loader(args.data_dir, pose_dir, audio_dir, is09_dir, args.batch_size, phase='test', shuffle=False, generator=generator)
    print('Test data loaded!')
    print('Finish loading the data....')

    torch.autograd.set_detect_anomaly(True)

    # Set model parameters
    args.n_class = 3  # Three-class task
    args.d_vin = 512  # Visual input dimension
    args.d_pose = 272  # Pose feature dimension
    args.d_aud = 768  # Audio feature dimension
    args.d_audio_in = 768
    args.d_audio_hidden = 128
    args.d_is09 = 384  
    args.dataset = dataset

    args.when = args.when  # Patience for adjusting learning rate
    args.criterion = 'CrossEntropyLoss'  # Loss function

    # Initialize the Solver class and perform training and evaluation
    solver = Solver(
        args,
        train_loader=train_loader,
        dev_loader=valid_loader,
        test_loader=test_loader,
        is_train=True,
        save_name=args.save_name  # Pass save_name to the solver
    )

    if torch.cuda.is_available():
        solver.model.to('cuda')

     solver.train_and_eval()


