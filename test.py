from zmq import device
import util, os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='Select device for training')
parser.add_argument('--dataset', type=str, default='METR-LA', help='Select dataset METR-LA/PEMS-BAY')
parser.add_argument('--data', type=str, default='./data/METR-LA', help='Select training data path')
parser.add_argument('--adjdata', type=str, default='./data/sensor_graph/adj_mat.pkl', help='Select the adjacency data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='Adjacency type')
parser.add_argument('--gcn_bool', action='store_true', help='Whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='Whether only adaptive adjacency')
parser.add_argument('--randomadj', action='store_true', help='Whether random initialize adaptive adjacency')
parser.add_argument('--seq_length', type=int, default=12, help='Select the sequence length')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=2, help='Select inputs dimension')
parser.add_argument('--num_nodes', type=int, default=207, help='Number of nodes')
parser.add_argument('--batch_size', type=int, default=64, help='Select batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Select learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='Select dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='Select weight decay rate')
parser.add_argument('--checkpoint', type=str, default="./checkpoints", help='Select the checkpoint directory')
parser.add_argument('--plotheatmap', type=str, default=True, help="Select the heatmap visualization mode")

args = parser.parse_args()

def main():
    device = torch.device(args.device)

    _, _, adj_mat = util.load_adj(args.adjdata, args.adjtype)
    supports = [torch.tensor(i).to(device) for i in adj_mat]
    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    model = GWNet(device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, aptinit=adjinit)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.checkpoint, args.dataset)))
    model.eval()

    print("Model Load Successfully")

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    amae, amape, armse = [], [], []

    for i in range(12):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = "Evaluate Best Model on test data for horizon {:3d}: Test MAE: {:.4f} | Test MAPE: {:.4f} | Test RMSE: {:.4f}"
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = "On average over 12 horizon: Test MAE: {:.4f} | Test MAPE: {:.4f} | Test RMSE: {:.4f}"
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))

    if args.plotheatmap:
        adp = F.softmax(F.relu(torch.mm(model.nodevec1, model.nodevec2)), dim=1)
        device = torch.device('cpu')
        adp.to(device)
        adp = adp.cpu().detach().numpy()
        adp = adp * (1/np.max(adp))
        df = pd.DataFrame(adp)
        sns.heatmap(df, cmap='RdYlBu')
        plt.savefig(f"./{args.checkpoint}/{args.dataset}/emb" + ".pdf")

    y12 = realy[:, 99, 11].cpu().detach().numpy()
    yhat12 = scaler.inverse_transform(yhat[:, 99, 11]).cpu().detach().numpy()

    y3 = realy[:, 99, 2].cpu().detach().numpy()
    yhat3 = scaler.inverse_transform(yhat[:, 99, 2]).cpu().detach().numpy()

    df2 = pd.DataFrame({'real12': y12, 'pred12': yhat12, 'real3': y3, 'pred3': yhat3})
    df2.to_csv(f".{args.checkpoint}/{args.dataset}/wave.csv", index=False)

if __name__ == "__main__":
    main()

# python test.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj