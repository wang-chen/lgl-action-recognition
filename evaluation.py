import tqdm
import torch


def performance(loader, net, device):
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(loader)):
            inputs, targets  = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()
        acc = correct/total
    return acc


if __name__ == "__main__":
    import argparse
    from ward import WARD
    import torch.utils.data as Data

    parser = argparse.ArgumentParser(description='Feature Graph Networks')
    parser.add_argument("--load", type=str, required=True, help="load pretrained model file")
    parser.add_argument("--device", type=str, default='cuda:0', help="cuda or cpu")
    parser.add_argument("--data-root", type=str, default='/data/datasets', help="dataset location to be download")
    parser.add_argument("--duration", type=int, default=50, help="duration")
    parser.add_argument("--batch-size", type=int, default=100, help="minibatch size")
    args = parser.parse_args(); print(args)

    test_data = WARD(root=args.data_root, duration=args.duration, train=False)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_data = WARD(root=args.data_root, duration=args.duration, train=True)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)

    net = torch.load(args.load)
    train_acc = performance(train_loader, net, args.device)
    test_acc = performance(test_loader, net, args.device)
    print("Evaluating model: ", args.load)
    print("Train Acc: %f; Test Acc: %f"%(train_acc, test_acc))
