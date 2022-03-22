import torch
from torch.optim import Adam
from ..base import add_base_train_args


def add_train_args(parser):
    group = parser.add_argument_group('Training')
    add_base_train_args(group)
    group.add_argument('--lr', type=float, default=1e-3)


def test(args, model, train_loader, test_loader):
    correct_train, correct_test = 0, 0
    with torch.no_grad():
        model.eval()
        for x, z, _ in train_loader:
            x, z = x.to(args.device), z.to(args.device)
            z_pred = model.classifier(x).argmax(dim=1)
            correct_train += z_pred.eq(z).sum().item()

        for x, z in test_loader:
            x, z = x.to(args.device), z.to(args.device)
            z_pred = model.classifier(x).argmax(dim=1)
            correct_test += z_pred.eq(z).sum().item()

        correct_train /= len(train_loader.dataset)
        correct_test /= len(test_loader.dataset)

        print(f'======> Inference acc: {correct_train * 100:.2f}')
        print(f'======>  Learning acc: {correct_test * 100:.2f}')


def train_epoch(args, epoch, model, train_loader, opt):
    loss_total = 0.0

    model.train()
    for x, z, z_mv in train_loader:
        model.zero_grad()
        x, z, z_mv = x.to(args.device), z.to(args.device), z_mv.to(args.device)
        loss = model(x, z_mv)
        loss.backward()
        opt.step()

        loss_total += loss.item()
    
    print(f'Epoch {epoch:4d} | Loss: {loss_total:.3f}')


def train(args, model, train_loader, test_loader):
    opt = Adam(model.classifier.parameters(), lr=args.lr)

    for e in range(1, args.n_epochs + 1):
        train_epoch(args, e, model, train_loader, opt)
        if e % args.test_interval == 0 or e == args.n_epochs:
            test(args, model, train_loader, test_loader)

