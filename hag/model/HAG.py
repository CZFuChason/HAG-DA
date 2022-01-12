import torch
import torch.nn as nn

from .SeqContext import SeqContext
from .EdgeAtt import EdgeAtt
from .GATCN import GATCN
from .GATCN import ResGATEGraph
from .Classifier import Classifier
from .functions import batch_graphify
import hag

log = hag.utils.get_logger()


class HAG(nn.Module):

    def __init__(self, args):
        super(HAG, self).__init__()
        u_dim = 768
        g_dim = 512
        h1_dim = 384 
        h2_dim = 200 
        hc_dim = 200
        tag_size = 16

        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device

        self.rnn = SeqContext(u_dim, g_dim, args)
        self.resgategraph = ResGATEGraph(g_dim, h1_dim, g_dim, args)
        self.edge_att = EdgeAtt(g_dim, args)
        self.gcn = GATCN(g_dim, h1_dim, h2_dim, args)
        self.clf = Classifier(g_dim + h2_dim, hc_dim, tag_size, args)

        edge_type_to_idx = {}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idx[str(j) + str(k) + '0'] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + '1'] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx
        log.debug(self.edge_type_to_idx)

    def get_rep(self, data):
        node_features = self.rnn(data["text_len_tensor"], data["text_tensor"]) # [batch_size, mx_len, D_g]
        features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(
            node_features, data["text_len_tensor"], data["speaker_tensor"], self.wp, self.wf,
            self.edge_type_to_idx, self.edge_att, self.device)

        
        graph_out = self.gcn(features, edge_index, edge_norm, edge_type)
        features = self.resgategraph(features, edge_index, edge_norm, edge_type) #v2
        
        return graph_out, features

    def forward(self, data):
        graph_out, features = self.get_rep(data)
        out = self.clf(torch.cat([features, graph_out], dim=-1), data["text_len_tensor"])

        return out

    def get_loss(self, data):
        graph_out, features = self.get_rep(data)
        loss = self.clf.get_loss(torch.cat([features, graph_out], dim=-1),
                                 data["label_tensor"], data["text_len_tensor"])

        return loss
