import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    

    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')

    parser.add_argument('--batch', type=int, default=8,
                        help='batch size')

    parser.add_argument('--model', type=str, default='convext', help='model name')
    
    parser.add_argument('--finetune', type=bool, default=False, help='finetune by adding layers')


    parser.add_argument('--num_classes', type=int, default=7, help="number \
                        of classes")

    parser.add_argument('--gpu', type=bool, default=True, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    
    parser.add_argument('--device', type=str, default='cuda:0', help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")

    parser.add_argument('--optimizer', type=str, default='adamx', help="type \
                        of optimizer")
    
    parser.add_argument('--modality', type=str, default='augmented', help="type of data [original or augmented or GAN]")
    
    parser.add_argument('--imbalanced', type=bool, default=False, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")

    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    
    parser.add_argument('--tensorboard', type=bool, default=True,
                        help='Log Metrics to TensorBoard')
    parser.add_argument('--logger', type=str, default = 'tb',
                       help= 'Logger / tensorboard(tb) or Wandb(wb)')

    args = parser.parse_args()

    return args