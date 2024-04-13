from argparse import ArgumentParser
import veri776
import Transforms
from termcolor import cprint
from model import make_model
from Trainer import ReIDTrainer
from torch.optim import SGD
import os
from torch.nn import CrossEntropyLoss, TripletMarginLoss
import DMT_scheduler
import torch



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../veri776')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch_size', '-b', type=int, default=16) # -b=20 is the limit on my machine
    parser.add_argument('--lr', '-l', type=float, default=1e-2)
    parser.add_argument('--epochs', '-e', type=int, default=50)
    parser.add_argument('--smoothing', type=float, default=0)
    parser.add_argument('--margin', '-m', type=float, default=0.6)
    parser.add_argument('--save_dir', '-s', type=str)
    parser.add_argument('--check_init', action='store_true')
    parser.add_argument('--backbone', type=str, choices=['resnet', 'resnext'])

    args = parser.parse_args()


    # datasets
    train_loader = veri776.get_veri776_train(
        veri776_path=args.dataset,
        num_workers=args.workers,
        batch_size=args.batch_size,
        transform=Transforms.get_training_transform(),
    )

    test_set = veri776.get_veri776_test(
        veri_776_path=args.dataset,
        transform=Transforms.get_test_transform(),
    )

    # neural network
    net = make_model(backbone=args.backbone, num_classes=576)
    print(net)



    # Trainer
    optim = SGD(net.parameters(), lr=args.lr)
    scheduler = DMT_scheduler.get_scheduler(optim)

    trainer = ReIDTrainer(
        net=net,
        ce_loss_fn=CrossEntropyLoss(label_smoothing=args.smoothing),
        triplet_loss_fn=TripletMarginLoss(margin=args.margin),
        optimizer=optim,
        lr_scheduler=scheduler,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    trainer.fit(
        train_loader=train_loader,
        test_set=test_set,
        gt_index_path=os.path.join(args.dataset, 'gt_index.txt'),
        name_query_path=os.path.join(args.dataset, 'name_query.txt'),
        jk_index_path=os.path.join(args.dataset, 'jk_index.txt'),
        epochs=args.epochs,
        save_dir=args.save_dir,
        check_init=args.check_init
    )
