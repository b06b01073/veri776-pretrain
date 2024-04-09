from tqdm import tqdm
import torch
import os
from einops import rearrange

class ReIDTrainer:
    '''
        A wrapper class that conducts the training process 
    '''
    def __init__(self, net, ce_loss_fn, triplet_loss_fn, optimizer, log_file=None, transform_scripts=None):
        '''
            Args: 
                net (nn.Module): the network to be trained 
                ce_loss_fn (CrossEntropyLoss): cross entropy    loss function from Pytorch
                triplet_loss_fn (TripletMarginLoss): triplet loss function from Pytorch
                optimizer (torch.optim): optimizer for `net`
                log_file (str): the file to store the intermediate performance 
        '''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = net.to(self.device)

        print(f'Training on {self.device}')


        self.ce_loss_fn = ce_loss_fn
        self.triplet_loss_fn = triplet_loss_fn

        self.optimizer = optimizer

        self.log_file = log_file

        self.transform_scripts = transform_scripts
    

    def fit(self, train_loader, epochs):
        '''
            Train the model for `epochs` epochs, where each epoch is composed of a training step and a validation step 

            Args:
                train_loader (DataLoader): a dataloader that wraps the training dataset (a ReIDDataset instance)
                gallery_loader (DataLoader): a dataloader that wraps the gallery (a ImageDataset instance)
                probes_loader (DataLoader): a dataloader that wraps the probes (a ImageDataset instance)
                epochs (int): epochs
                save_dir (str): the path to save the model
                topk (list of int): the list of top k value for rank k evaluation
                k (int): k in k nearest neighbor
        '''

        for epoch in range(epochs):
            self.train(train_loader)
            # result = self.val(gallery_loader, probes_loader, topk)


    def train(self, train_loader):
        '''
            Args:
                train_loader (DataLoader): a dataloader that wraps the training dataset (a ReIDDataset instance)
        '''
        self.net.train()

        total_ce_loss = 0
        total_triplet_loss = 0
        for images, labels in tqdm(train_loader, dynamic_ncols=True, desc='train'):
            images, labels = images.to(self.device), labels.to(self.device)
            
            anchors = images[:, 0, :].squeeze()
            positvies = images[:, 1, :].squeeze()
            negatives = images[:, 2, :].squeeze()

            anchor_embeddings, _, anchor_out = self.net(anchors)
            positvie_embeddings, _, positive_out = self.net(positvies)
            negative_embeddings, _, negative_out = self.net(negatives)

            triplet_loss = self.triplet_loss_fn(anchor_embeddings, positvie_embeddings, negative_embeddings)

            
            preds = rearrange([anchor_out, positive_out, negative_out], 't b e -> (b t) e')
            labels_batch = torch.flatten(labels)

            ce_loss = self.ce_loss_fn(preds, labels_batch)


            self.optimizer.zero_grad
            loss = triplet_loss + ce_loss
            loss.backward()
            self.optimizer.step()

            total_ce_loss += ce_loss.item()
            total_triplet_loss += triplet_loss.item()

        print(f'triplet_loss: {total_triplet_loss:.4f}, ce_loss: {total_ce_loss:.4f}')
