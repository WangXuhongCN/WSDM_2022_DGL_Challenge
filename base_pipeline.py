import torch.nn.functional as F
import torch
import dgl
import dgl.nn.pytorch as dglnn
from torch import nn
import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def get_args():
    ### Argument and global variables
    parser = argparse.ArgumentParser('csv2DGLgraph')
    parser.add_argument('--dataset', type=str, choices=["A", "B"], default = 'B', help='Dataset name')
    parser.add_argument("--lr", type=float, default=1e-2,help="learning rate")
    parser.add_argument('--epochs', type=int, default = 500, help='Number of epochs')
    parser.add_argument("--emb_dim", type=int, default=16,help="number of hidden gnn units")
    parser.add_argument("--n_layers", type=int, default=2,help="number of hidden gnn layers")
    parser.add_argument("--gpu", type=int, default=0,help="GPU id, -1 means using CPU")
    parser.add_argument("--weight_decay", type=float, default=5e-4,help="Weight for L2 loss")

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args

class hetero_conv(nn.Module):
    def __init__(self, etypes, n_layers, in_feats, hid_feats, activation, dropout=0.2):
        super(hetero_conv, self).__init__()
        self.etypes = etypes
        self.n_layers = n_layers
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.act = activation
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.hconv_layers = nn.ModuleList()

        # input layer
        self.hconv_layers.append(self.build_hconv(in_feats,hid_feats,activation=self.act))
        # hidden layers
        for i in range(n_layers - 1):
            self.hconv_layers.append(self.build_hconv(hid_feats,hid_feats,activation=self.act))
        # output layer
        self.hconv_layers.append(self.build_hconv(hid_feats,hid_feats)) # activation None

        self.fc1 = nn.Linear(hid_feats*2, hid_feats)
        self.fc2 = nn.Linear(hid_feats, 10)
    
    def build_hconv(self,in_feats,out_feats,activation=None):
        GNN_dict = {}
        for event_type in self.etypes:
            GNN_dict[event_type] = dglnn.SAGEConv(in_feats=in_feats,out_feats=out_feats,aggregator_type='mean',activation=activation) 
        return dglnn.HeteroGraphConv(GNN_dict, aggregate='sum')

    def forward(self, g, feat_key='feat'):
        h = g.ndata[feat_key]
        if not isinstance(h,dict):
            h = {'Node':g.ndata[feat_key]}
        for i, layer in enumerate(self.hconv_layers):
            h = layer(g, h)
        return h

    def emb_concat(self, g, etype):
        def cat(edges):
            return {'emb_cat': torch.cat([edges.src['emb'], edges.dst['emb']],1)}
        with g.local_scope():
            g.apply_edges(cat, etype=etype)
            emb_cat = g.edges[etype].data['emb_cat']
        return emb_cat
    
    def time_predict(self, emb_cat):
        h = self.dropout(emb_cat)
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        h = h.sigmoid()
        return h

@torch.no_grad()
def int2dec_bits(x, bits=10): 
    inp = x.repeat(10,1).transpose(0,1)
    div = torch.cat([torch.ones((x.shape[0],1))*10**(bits-1-i) for i in range(bits)],1)
    return ((inp/div).int()%10).float()

@torch.no_grad()
def dec_bits2int(x, bits=10): 
    div = torch.cat([torch.ones((x.shape[0],1))*10**(bits-1-i) for i in range(bits)],1)
    return (x*div).sum(1)

def train(args, g):
    if args.gpu < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % args.gpu)

    if args.dataset == 'B':
        dim_nfeat = args.emb_dim*2 
        for ntype in g.ntypes:
            g.nodes[ntype].data['feat'] = torch.randn((g.number_of_nodes(ntype), dim_nfeat)) 
    else:
        dim_nfeat = g.ndata['feat'].shape[1]

    model = hetero_conv(g.etypes, args.n_layers, dim_nfeat, args.emb_dim, F.relu)

    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    loss_fcn = nn.BCELoss()  
    loss_values = []

    for i in range(args.epochs):
        model.train()
        node_emb = model(g)
        loss = 0
        for ntype in g.ntypes:
            g.nodes[ntype].data['emb'] = node_emb[ntype]
        for i, etype in enumerate(g.etypes):
            if etype.split('_')[-1] == 'reversed':
                continue
            emb_cat = model.emb_concat(g, etype)
            time_pred = model.time_predict(emb_cat)
            time_GT = int2dec_bits(g.edges[etype].data['ts'].int())*0.1
            loss += loss_fcn(time_pred, time_GT)
                
        loss_values.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Loss:', np.mean(loss_values))
        torch.cuda.empty_cache()
        # test every epoch
        test(args, g, model)
    return g, model
    
    
def preprocess(args, directed_g):
    # add reverse edges for model computing
    if args.dataset == 'A':
        g = dgl.add_reverse_edges(directed_g, copy_edata=True)
    if args.dataset == 'B':
        graph_dict = {}
        for (src_type, event_type, dst_type) in directed_g.canonical_etypes:
            graph_dict[(src_type, event_type, dst_type)] = directed_g.edges(etype = (src_type, event_type, dst_type))
            src_nodes_reversed = directed_g.edges(etype = (src_type, event_type, dst_type))[1]
            dst_nodes_reversed = directed_g.edges(etype = (src_type, event_type, dst_type))[0]
            graph_dict[(dst_type, event_type+'_reversed', src_type)] = (src_nodes_reversed,dst_nodes_reversed)
            #ts_dict[canonical_etype] = g.edges[canonical_etype].data['ts']
        g = dgl.heterograph(graph_dict)
        for etype in g.etypes:
            g.edges[etype].data['ts'] = directed_g.edges[etype.split('_')[0]].data['ts']
            if 'feat' in directed_g.edges[etype.split('_')[0]].data.keys():
                g.edges[etype].data['feat'] = directed_g.edges[etype.split('_')[0]].data['feat']
    return g

@torch.no_grad()
def test(args, g, model):
    model.eval()
    data_path = 'test_csvs/toy/'
    test_csv = pd.read_csv(data_path + f'Dataset_{args.dataset}_test_toy.csv',delimiter='\t')
    label = test_csv.exist.values
    src = test_csv.src.values
    dst = test_csv.dst.values
    start_at = test_csv.start_at.values
    end_at = test_csv.end_at.values
    if args.dataset == 'A':
        emb_cats =  torch.cat([g.ndata['emb'][src],g.ndata['emb'][dst]], 1)
    if args.dataset == 'B':
        emb_cats =  torch.cat([g.ndata['emb']['User'][src],g.ndata['emb']['Item'][dst]], 1)
    time_pred = model.time_predict(emb_cats).int()
    unix_time = dec_bits2int(time_pred*10).numpy()
    pred = (unix_time >= start_at) & (unix_time <= end_at)
    AUC = roc_auc_score(label,pred)
    print(f'AUC is {round(AUC,3)}')

if __name__ == "__main__":
    args = get_args()
    g = dgl.load_graphs(f'DGLgraphs/Dataset_{args.dataset}.bin')[0][0]
    g = preprocess(args, g)
    g, model = train(args,g)
    test(args, g, model)