import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .common import MLP
from .gcn import GCN, GCNII
import scipy.sparse as sp
from itertools import product


def adj_to_edges(adj):
    # Adj sparse matrix to list of edges
    rows, cols = np.nonzero(adj)
    edges = list(zip(rows.tolist(), cols.tolist()))
    return edges


def edges_to_adj(edges, n):
    # List of edges to Adj sparse matrix
    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype='float32')
    return adj


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GraphFull(nn.Module):
    def __init__(self, dset, args, element_embedding):
        super(GraphFull, self).__init__()
        self.args = args
        self.dset = dset

        self.val_forward = self.val_forward_dotpr
        self.train_forward = self.train_forward_normal
        # Image Embedder
        self.num_attrs, self.num_objs, self.num_pairs = len(dset.attrs), len(dset.objs), len(dset.all_pairs)
        # self.full_pairs = list(product(dset.attrs, dset.objs))
        self.pairs = dset.all_pairs

        all_element_words = list(self.dset.attrs) + list(self.dset.objs)
        self.attr_obj_displacement = len(dset.attrs)
        self.element_pair_displacement = len(all_element_words)

        self.dict_Obj2IDX = {word: idx for idx, word in enumerate(self.dset.objs)}
        self.dict_Attr2IDX = {word: idx for idx, word in enumerate(self.dset.attrs)}

        # self.embeddings = element_embedding
        if args.graph_init is not None:
            path = args.graph_init
            graph = torch.load(path)
            embeddings = graph['embeddings'].to(device)
            adj_matrix = graph['adj']
        else:
            # embeddings = self.init_embeddings(element_embedding).to(device) # element + pair embeddings
            # self.embeddings = embeddings
            adj_matrix = self.adj_from_pairs()
        # self.embeddings = embeddings


        hidden_layers = self.args.graph_gr_emb
        if args.graph_gcn_type == 'gcn':
            self.gcn = GCN(adj_matrix, args.graph_emb_dim, args.graph_emb_dim, hidden_layers)
        else:
            self.gcn = GCNII(adj_matrix, args.graph_emb_dim, args.graph_emb_dim, args.hidden_dim, args.gcn_nlayers, lamda = 0.5, alpha = 0.1, variant = False)


    def init_embeddings(self, element_embedding):

        def get_compositional_embeddings(embeddings, pairs):
            # Getting compositional embeddings from base embeddings
            composition_embeds = []
            for (attr, obj) in pairs:
                attr_embed = embeddings[self.dict_Attr2IDX[attr]]
                obj_embed = embeddings[self.dict_Obj2IDX[obj] + self.attr_obj_displacement]
                composed_embed = (attr_embed + obj_embed) / 2
                composition_embeds.append(composed_embed)
            composition_embeds = torch.stack(composition_embeds)
            print('Compositional Embeddings are ', composition_embeds.shape)
            return composition_embeds

        # init with word embeddings
        composition_embeds = get_compositional_embeddings(element_embedding, self.pairs)
        full_embeddings = torch.cat([element_embedding, composition_embeds], dim=0)

        return full_embeddings


    def update_dict(self, wdict, row,col,data):
        wdict['row'].append(row)
        wdict['col'].append(col)
        wdict['data'].append(data)

    def adj_from_pairs(self):

        def edges_from_pairs_close_world(pairs):
            weight_dict = {'data':[],'row':[],'col':[]}


            for i in range(self.element_pair_displacement):
                self.update_dict(weight_dict,i,i,1.)

            for idx, (attr, obj) in enumerate(pairs):
                attr_idx, obj_idx = self.dict_Attr2IDX[attr], self.dict_Obj2IDX[obj] + self.num_attrs

                self.update_dict(weight_dict, attr_idx, obj_idx, 1.)
                self.update_dict(weight_dict, obj_idx, attr_idx, 1.)

                pair_node_id = idx + self.element_pair_displacement
                self.update_dict(weight_dict,pair_node_id,pair_node_id,1.)

                self.update_dict(weight_dict, pair_node_id, attr_idx, 1.)
                self.update_dict(weight_dict, pair_node_id, obj_idx, 1.)


                self.update_dict(weight_dict, attr_idx, pair_node_id, 1.)
                self.update_dict(weight_dict, obj_idx, pair_node_id, 1.)

            return weight_dict

        def edges_from_pairs_open_world(pairs):

            # result
            weight_dict = {'data': [], 'row': [], 'col': []}

            # load the feasible scores
            import os
            feasibility_path = os.path.join(f"{self.args.dataset_dir}", f'feasibility_{self.args.dataset}.pt')
            feasibility_score = torch.load(feasibility_path, map_location='cpu')['feasibility']

            # check feasibility_score
            # sel_pair = ('ancient','building')
            # sel_pair_idx = self.dset.allPair2idx[sel_pair]
            # print(feasibility_score[sel_pair_idx])

            # -------------------------------
            # check unfeasible pairs
            # 1. mit-states: only 2 unfeasible
            #   ('blunt', 'beach')
            #   ('closed', 'persimmon')
            # --------------------------------
            # for idx, item in enumerate(feasibility_score):
            #    if item < 0:
            #         print(pairs[idx])

            # add element self-cycle
            for i in range(self.element_pair_displacement):
                self.update_dict(weight_dict, i, i, 1.)

            for pair_idx, (attr, obj) in enumerate(pairs):

                pair_node_id = pair_idx + self.element_pair_displacement
                attr_idx, obj_idx = self.dict_Attr2IDX[attr], self.dict_Obj2IDX[obj] + self.num_attrs

                if (attr, obj) in self.dset.train_pairs:
                    original_score = feasibility_score[pair_idx]
                    score = 1.0
                else:
                    score = max(0., feasibility_score[pair_idx])

                self.update_dict(weight_dict, attr_idx, obj_idx, score)
                self.update_dict(weight_dict, obj_idx, attr_idx, score)

                # add pair node self-cycle
                self.update_dict(weight_dict, pair_node_id, pair_node_id, 1.)

                #  pair --> element is 1;
                self.update_dict(weight_dict, pair_node_id, attr_idx, 1.)
                self.update_dict(weight_dict, pair_node_id, obj_idx, 1.)

                # element --> pair is feasibility
                self.update_dict(weight_dict, attr_idx, pair_node_id, score)
                self.update_dict(weight_dict, obj_idx, pair_node_id, score)



            return weight_dict

        if self.args.open_world:
            edges = edges_from_pairs_open_world(self.pairs)
        else:
            edges = edges_from_pairs_close_world(self.pairs)
        adj = sp.csr_matrix((edges['data'], (edges['row'], edges['col'])),
                            shape=(len(self.pairs) + self.element_pair_displacement, len(self.pairs) + self.element_pair_displacement))

        return adj



    def train_forward_normal(self, embeddings):
        current_embeddings = self.gcn(embeddings)
        return  current_embeddings

    def val_forward_dotpr(self, x):
        img = x[0]
        batch_size = img.shape[0]

        if self.args.nlayers:
            img_feats = self.image_embedder(img)
        else:
            img_feats = (img)


        if self.cosloss>=0:
            img_feats = F.normalize(img_feats, dim=1)


        current_embedddings = self.gcn(self.embeddings())

        pair_embeds = current_embedddings[self.num_attrs+self.num_objs:self.num_attrs+self.num_objs+self.num_pairs,:].permute(1,0)

        score = torch.matmul(img_feats, pair_embeds)

        scores = {}
        for itr, pair in enumerate(self.dset.full_pairs):
            scores[pair] = score[:,self.dset.all_pair2idx[pair]]
        
        return score, scores
        
    def val_forward_distance_fast(self, x):
        img = x[0]

        img_feats = (self.image_embedder(img))
        current_embeddings = self.gcn(self.embeddings)
        pair_embeds = current_embeddings[self.num_attrs+self.num_objs:,:]

        batch_size, pairs, features = img_feats.shape[0], pair_embeds.shape[0], pair_embeds.shape[1]
        img_feats = img_feats[:,None,:].expand(-1, pairs, -1)
        pair_embeds = pair_embeds[None,:,:].expand(batch_size, -1, -1)
        diff = (img_feats - pair_embeds)**2
        score = diff.sum(2) * -1

        scores = {}
        for itr, pair in enumerate(self.dset.full_pairs):
            scores[pair] = score[:,self.dset.all_pair2idx[pair]]

        return None, scores

    def forward(self, embeddings):
        # if self.training:
        #    current_embedding = self.train_forward(embeddings)
        # else:
        #    with torch.no_grad():
        #        loss, pred = self.val_forward(x)
        current_embedding = self.train_forward(embeddings)
        return current_embedding