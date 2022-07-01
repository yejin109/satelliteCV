import os
import yaml
import numpy as np
from tqdm import tqdm
from itertools import product

import torch.cuda
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from util.util_data import ODDataset, collate_fn
from util.model import load_model, load_mobilenet_large, load_mobilenet_large_320, load_resnet


def train(model, optimizer, loader):
    print('Training')

    # initialize tqdm progress bar
    prog_bar = tqdm(loader, total=len(loader))
    loss_cls = []
    loss_box = []
    loss_obj = []
    loss_rpn_box = []
    loss = []
    for epoch in range(args['epochs']):
        for i, data in enumerate(prog_bar):
            optimizer.zero_grad()
            images, targets = data

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            loss.append(loss_value)

            loss_cls.append(loss_dict['loss_classifier'].detach().cpu().numpy())
            loss_box.append(loss_dict['loss_box_reg'].detach().cpu().numpy())
            loss_obj.append(loss_dict['loss_objectness'].detach().cpu().numpy())
            loss_rpn_box.append(loss_dict['loss_rpn_box_reg'].detach().cpu().numpy())
            losses.backward()
            optimizer.step()
            # update the loss value beside the progress bar for each iteration
            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
        if epoch % 5:
            torch.save(model.state_dict(), f"snapshot/model/{model_type}_{args['lr']}_ {epoch}th_epoch.pt")
        print(f"Epoch {epoch+1} is DONE")
    return loss, loss_cls, loss_box, loss_obj, loss_rpn_box


def main(model):
    transformer = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = ODDataset(path_labels, path_objects, args['raw_size'], args['post_size'], transformer)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [args['train_size'],
                                                                           args['val_size'],
                                                                           args['test_size']])
    train_loader = DataLoader(
        train_set,
        batch_size=args['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    # model = load_model(args['num_classes']).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    losses = train(model, optimizer, train_loader)
    np.savez(f"snapshot/loss/train_loss_{model_type}_{args['lr']}.npz", total=np.array(losses[0]), cls=np.array(losses[1]), box=np.array(losses[2])
             , objectness=np.array(losses[3]), rpn_box=np.array(losses[4]))
    model = None
    train_loader = None


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    path_root = os.getcwd()
    path_objects = f"{path_root}/data/raw/objects"
    path_labels = f"{path_root}/data/raw/labels"
    objects = os.listdir(path_objects)
    labels = os.listdir(path_labels)

    with open('.config.yaml', 'r') as f:
        args = yaml.safe_load(f)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for i in range(2):
        if i == 0:
            model_type = 'mobile_pure'
            model = load_mobilenet_large(args['num_classes']).to(device)
            for lr_case in [0.01, 0.001]:
                args['lr'] = lr_case
                main(model)
        else:
            model_type = 'resnet'
            model = load_mobilenet_large(args['num_classes']).to(device)
            for lr_case in [0.01, 0.001]:
                args['lr'] = lr_case
                main(model)

