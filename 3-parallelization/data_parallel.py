import csv
from mpi4py import MPI
import numpy as np
import MNISTh as mh
from MNISTh.ConvNet import ConvNet
import copy
import torch
from torch import nn
import time
import random


def main(trainloader, lr = 1e-3, criterion = nn.CrossEntropyLoss(), print_every=10):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_nodes = comm.Get_size()
    
    # seed = seed or int(time.time()*1e6 % 100000)
    # random.seed(seed) 
    # np.random.seed(seed) 
    # torch.manual_seed(seed)
    
    if rank == 0:
        filename = "parallel_results.csv"
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["i", "sum_loss"])
        model = ConvNet()
        optimizer = torch.optim.Adam(model.parameters(), lr)
        
    for i, (all_inputs, all_labels) in enumerate(trainloader, 0):
        
        if rank == 0:
            optimizer.zero_grad()
            inputs, labels = ( 
                torch.chunk(all_inputs, chunks = n_nodes, dim = 0), 
                torch.chunk(all_labels, chunks = n_nodes, dim = 0)
            )
            split_models = [ copy.deepcopy(model) for _ in range(n_nodes) ]
        else:
            inputs, labels, split_models = None, None, None

        inputs_scatter = comm.scatter(inputs, root = 0)
        labels_scatter = comm.scatter(labels, root = 0)
        models_scatter = comm.scatter(split_models, root = 0)

        outputs = models_scatter(inputs_scatter)
        losses = criterion(outputs, labels_scatter)
        losses.backward()
        
        grads = { name: param.grad for name, param in models_scatter.named_parameters() }
        all_grads = comm.gather(grads, root = 0)
        all_loss = comm.gather(losses, root = 0)
        if rank == 0:
            sum_grads = {}
            for key in all_grads[0]:
                get_grad_list = [ this_grad[key] for this_grad in all_grads ] 
                sum_grads[key] = sum(get_grad_list)

            for name, param in model.named_parameters():
                param.grad = sum_grads[name]
                
            optimizer.step()

            if i % print_every == print_every - 1:
                mean_loss = np.mean([ l.detach().numpy() for l in all_loss ])
                with open(filename, 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([i, mean_loss])

        
if __name__ == '__main__':
    trainloader = mh.trainloader
    main(trainloader)