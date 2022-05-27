import os, time
import util, torch
import numpy as np
import argparse
from engine import Trainer
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='Select the training device')
parser.add_argument('--dataset', type=str, default='METR-LA', help='Select dataset METR-LA/PEMS-BAY')
parser.add_argument('--data', type=str, default='./data/METR-LA', help='Select training data path')
parser.add_argument('--adjdata', type=str, default='./data/sensor_graph/adj_mx.pkl', help='Select the adjacency data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='Adjacency type')
parser.add_argument('--gcn_bool', action='store_true', help='Whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='Whether only adaptive adjacency')
parser.add_argument('--addaptadj', action='store_true', help='Whether add adaptive adjacency')
parser.add_argument('--randomadj', action='store_true', help='Whether random initialize adaptive adjacency')
parser.add_argument('--seq_length', type=int, default=12, help='Select the sequence length')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=2, help='Select inputs dimension')
parser.add_argument('--num_nodes', type=int, default=207, help='Number of nodes')
parser.add_argument('--batch_size', type=int, default=4, help='Select batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Select learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='Select dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='Select weight decay rate')
parser.add_argument('--epoch', type=int, default=3, help='Number of training epochs')
parser.add_argument('--print_every', type=int, default=50, help="Show verbose results every setting iteration")
parser.add_argument('--checkpoint', type=str, default="./checkpoints", help='Select the checkpoint directory')
parser.add_argument('--exp_id', type=int, default=1, help='Experiment ID')



def main():
    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)

    os.makedirs(os.path.join(args.checkpoint, args.dataset), exist_ok=True)

    # Load data
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_idx, adj_mat = util.load_adj(args.adjdata, args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mat]
    
    print("Training Arguments: \n", args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    engine = Trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,\
                    args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj, adjinit)

    print("Start Training ...", flush=True)
    loss_list = []
    train_time, val_time = [], []
    for i in range(1, args.epoch+1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device).transpose(1, 3)
            trainy = torch.Tensor(y).to(device).transpose(1, 3)
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])

            if iter % args.print_every == 0:
                log = "Iter: {:03d} | Train Loss: {:.4f} | Train MAPE: {:.4f} | Train RMSE: {:.4f}"
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)

        engine.scheduler.step()
        t2 = time.time()
        train_time.append(t2-t1)

        # Validation
        val_loss = []
        val_mape = []
        val_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            valx = torch.Tensor(x).to(device).transpose(1, 3)
            valy = torch.Tensor(y).to(device).transpose(1, 3)
            metrics = engine.eval(valx, valy[:, 0, :, :])
            val_loss.append(metrics[0])
            val_mape.append(metrics[1])
            val_rmse.append(metrics[2])

        s2 = time.time()
        log = "Epoch: {:03d}, Inference Time: {:.4f} secs"
        print(log.format(i, (s2 - s1)))
        val_time.append(s2-s1)
        mean_train_loss = np.mean(train_loss)
        mean_train_mape = np.mean(train_mape)
        mean_train_rmse = np.mean(train_rmse)

        mean_val_loss   = np.mean(val_loss)
        mean_val_mape   = np.mean(val_mape)
        mean_val_rmse   = np.mean(val_rmse)
        loss_list.append(mean_val_loss)

        log = "Epoch: {:03d} | Train Loss: {:.4f} | Train MAPE: {:.4f} | Train RMSE: {:.4f} | Val Loss: {:.4f} | Val MAPE: {:.4f} | Val RMSE: {:.4f}"
        print(log.format(i, mean_train_loss, mean_train_mape, mean_train_rmse, mean_val_loss, mean_val_mape, mean_val_rmse), flush=True)
        torch.save(engine.model.state_dict(), os.path.join(args.checkpoint, args.dataset, f"epoch_{str(i)}_{str(round(mean_val_loss, 2))}.pth"))
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # Testing
    best_id = np.argmin(loss_list)
    engine.model.load_state_dict(torch.load(os.path.join(args.checkpoint, args.dataset, f"epoch_{str(best_id+1)}_{str(round(loss_list[best_id], 2))}.pth")))

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())
    
    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    print("Training Finished")
    print("The valid loss on best model is: ", str(round(loss_list[best_id], 4)))

    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = "Evaluate best model on test data for horizon {:d} | Test MAE: {:.4f} | Test MAPE: {:.4f} | Test RMSE: {:.4f}"
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = "On average over 12 horizons | Test MAE: {:.4f} | Test MAPE: {:.4f} | Test RMSE: {:.4f}"
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
    torch.save(engine.model.state_dict(), os.path.join(args.checkpoint, args.dataset, f"exp_{args.exp_id}_best_loss_{str(round(loss_list[best_id], 2))}.pth"))


if __name__ == "__main__":
    args = parser.parse_args()
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))

# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj