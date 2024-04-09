from argparse import ArgumentParser
import veri776
import Transforms
from termcolor import cprint
from model import Resnet101IbnA
from Trainer import ReIDTrainer
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, TripletMarginLoss

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../veri776')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--embedding_dim', type=int, default=2048)
    parser.add_argument('--lr', '-l', type=float, default=3.5e-4)
    parser.add_argument('--epochs', '-e', type=int, default=120)
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--margin', '-m', type=float, default=0.6)

    args = parser.parse_args()

    train_loader, test_loader = veri776.get_veri776(
        veri776_path=args.dataset,
        num_workers=args.workers,
        batch_size=args.batch_size,
        training_transform=Transforms.get_training_transform(),
        test_transform=Transforms.get_test_transform(),
    )

       # neural network
    net = Resnet101IbnA(embedding_dim=args.embedding_dim, num_classes=576)
    
    cprint('Your model:', color='green')
    print(net)

    freezed_layers = ['layer1', 'layer2', 'layer3']
    for name, param in net.named_parameters():
        for layer in freezed_layers:
            if layer in name:
                param.requires_grad = False



    # Trainer
    trainer = ReIDTrainer(
        net=net,
        ce_loss_fn=CrossEntropyLoss(label_smoothing=args.smoothing),
        triplet_loss_fn=TripletMarginLoss(margin=args.margin),
        optimizer=Adam(net.parameters(), lr=args.lr),
    )




    trainer.fit(
        train_loader=train_loader,
        epochs=args.epochs,
    )