import numpy as np
import torch
from model import SessionGraph
import argparse


class Predictor():
    def __init__(self, opt, device):
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device

        self.n_node = 43098
        if opt.dataset == 'diginetica':
            self.n_node = 43098
            self.len_max = 69
        elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
            self.n_node = 37484
            self.len_max = 70 # 未知
        else:
            self.n_node = 310
            self.len_max = 70  # 未知
        # self.model = SessionGraph(self.opt, self.n_node)
        self.model = torch.load(opt.checkpoint)
        self.model = self.model.to(device)
        self.model.eval()

    def data_preprocess(self, sessions):
        us_lens = [len(upois) for upois in sessions]
        inputs = [upois +[0] * (self.len_max - le) for upois, le in zip(sessions, us_lens)]
        mask = [[1] * le + [0] * (self.len_max - le) for le in us_lens]
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])

        return torch.Tensor(alias_inputs).long(), torch.Tensor(A).float(), torch.Tensor(items).long(), torch.Tensor(mask).long()

    def predict(self, sessions):
        with torch.no_grad():
            alias_inputs, A, items, mask = self.data_preprocess(sessions)
            alias_inputs = alias_inputs.to(device)
            mask = mask.to(device)
            items = items.to(self.device)
            A = A.to(self.device)

            hidden = self.model(items, A)  # 跳到98行

            get = lambda i: hidden[i][alias_inputs[i]]

            seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
            out = self.model.compute_scores(seq_hidden, mask)
            next_item_id = out.argmax(dim=1)
            p = out.max(dim=1)
            return next_item_id.item(),p[0].item()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='diginetica',
                        help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
    parser.add_argument('--checkpoint', default="./checkpoints/sr-gnn/best_hit.pt", help="checkpoint path")
    parser.add_argument('--batchSize', type=int, default=1024, help='input batch size')
    parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
    parser.add_argument('--epoch', type=int, default=100, help='the number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--lr_dc_step', type=int, default=3,
                        help='the number of steps after which the learning rate decay')
    parser.add_argument('--l2', type=float, default=1e-5,
                        help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
    parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
    parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
    parser.add_argument('--validation', action='store_true', help='validation')
    parser.add_argument('--valid_portion', type=float, default=0.1,
                        help='split the portion of training set as validation set')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = Predictor(opt,device)

    session = [[5,6]]
    next_item_id,p  = model.predict(session)
    print(next_item_id,p)