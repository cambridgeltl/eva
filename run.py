#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import time
import argparse
import gc
import random
import math
import numpy as np
import scipy
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from models import *
from Load import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", type=str, default="data/DBP15K/zh_en", required=False, help="input dataset file directory, ('data/DBP15K/zh_en', 'data/DWY100K/dbp_wd')")
    parser.add_argument("--rate", type=float, default=0.3, help="training set rate")

    parser.add_argument("--save", default="", help="the output dictionary of the model and embedding. (should be created manually)")
    parser.add_argument("--cuda", action="store_true", default=True, help="whether to use cuda or not")

    parser.add_argument("--seed", type=int, default=2021, help="random seed")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs to train")
    parser.add_argument("--check_point", type=int, default=100, help="check point")

    parser.add_argument("--hidden_units", type=str, default="128,128,128", help="hidden units in each hidden layer(including in_dim and out_dim), splitted with comma")
    parser.add_argument("--heads", type=str, default="2,2", help="heads in each gat layer, splitted with comma")
    parser.add_argument("--instance_normalization", action="store_true", default=False, help="enable instance normalization")

    parser.add_argument("--lr", type=float, default=0.005, help="initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay (L2 loss on parameters)")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate for layers")
    parser.add_argument("--attn_dropout", type=float, default=0.0, help="dropout rate for gat layers")
    parser.add_argument("--dist", type=int, default=2, help="L1 distance or L2 distance. ('1', '2')")

    parser.add_argument("--margin_CG", type=int, default=3, help="margin for cross-graph model")
    parser.add_argument("--k_CG", type=int, default=25, help="negtive sampling number for cross-graph model")
    parser.add_argument("--update_num", type=int, default=5, help="number of epoch for updating negtive samples")

    parser.add_argument("--wo_K", action="store_true", default=False, help="baseline w/o Knowledge embedding model")
    parser.add_argument("--wo_NNS", action="store_true", default=False, help="baseline w/o NNS")

    parser.add_argument("--csls", action="store_true", default=False, help="use CSLS for inference")
    parser.add_argument("--csls_k", type=int, default=10, help="top k for csls")

    parser.add_argument("--il", action="store_true", default=False, help="Iterative learning?")
    parser.add_argument("--semi_learn_step", type=int, default=10, help="If IL, what's the update step?")
    parser.add_argument("--il_start", type=int, default=500, help="If Il, when to start?")

    parser.add_argument("--bsize", type=int, default=7500, help="batch size")

    parser.add_argument("--unsup", action="store_true", default=False)
    parser.add_argument("--unsup_k", type=int, default=1000, help="|visual seed|")

    #parser.add_argument("--long_tail_analysis", action="store_true", default=False)
    parser.add_argument("--lta_split", type=int, default=0, help="split in {0,1,2,3,|splits|-1}")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    K_CG = args.k_CG

    # Load data
    lang_list = [1, 2]
    ent2id_dict, ills, triples, r_hs, r_ts, ids = read_raw_data(args.file_dir, lang_list)
    e1 = os.path.join(args.file_dir,'ent_ids_1')
    e2 = os.path.join(args.file_dir,'ent_ids_2')
    left_ents = get_ids(e1)
    right_ents = get_ids(e2)

    ENT_NUM = len(ent2id_dict)
    REL_NUM = len(r_hs)
    
    np.random.shuffle(ills)

    # FR-EN
    #img_vec_path = "/mnt/HDD_4T/dbpedia/fr_en_GA_id_img_feature_dict.pkl"
    #img_vec_path = "/mnt/HDD_4T/dbpedia/fr_en_GA_id_img_feature_dict_infobox_included.pkl"
    #img_vec_path = "/mnt/HDD_4T/dbpedia/fr_en_GA_id_img_feature_dict_infobox_only.pkl"
    #img_vec_path = "/mnt/HDD_4T/dbpedia/fr_en_GA_id_img_feature_dict_en_infobox_included_fr_infobox_only.pkl"

    if "V1" in args.file_dir:
        split = "norm"
        img_vec_path = "data/pkls/dbpedia_wikidata_15k_norm_GA_id_img_feature_dict.pkl"
    elif "V2" in args.file_dir:
        split = "dense"
        img_vec_path = "data/pkls/dbpedia_wikidata_15k_dense_GA_id_img_feature_dict.pkl"
    else:    
        split = args.file_dir.split("/")[-1]
        img_vec_path = "data/pkls/"+ split+ "_GA_id_img_feature_dict.pkl"
    
    img_features = load_img(ENT_NUM, img_vec_path)

    img_features = F.normalize(torch.Tensor(img_features).to(device))
    print ("image feature shape:", img_features.shape)

    weight_raw = torch.tensor([1.0,1.0,1.0,1.0], requires_grad=True, device=device)

    # if unsupervised? use image to obtain links
    #"""
    if args.unsup:
        l_img_f = img_features[left_ents] # left images
        r_img_f = img_features[right_ents] # right images

        #print (l_img_f.shape, r_img_f.shape)
        img_sim = l_img_f.mm(r_img_f.t())
        #print (img_sim.shape)
        topk = args.unsup_k
        two_d_indices = get_topk_indices(img_sim, topk*100)
        del l_img_f, r_img_f, img_sim

        visual_links = []
        used_inds = []
        count = 0
        for ind in two_d_indices:
            if left_ents[ind[0]] in used_inds: continue
            if right_ents[ind[1]] in used_inds: continue
            used_inds.append(left_ents[ind[0]])
            used_inds.append(right_ents[ind[1]])
            visual_links.append((left_ents[ind[0]], right_ents[ind[1]]))
            count += 1
            if count == topk: 
                break

        #"""
        count = 0.0
        for link in visual_links: 
            if link in ills:
                count = count + 1
        print ("%.2f%% in true links" % (count/len(visual_links)*100))
        print ("visual links length: %d" % (len(visual_links)))
        #exit()
        train_ill = np.array(visual_links, dtype=np.int32)
    else:
        # if supervised
        train_ill = np.array(ills[:int(len(ills) // 1 * args.rate)], dtype=np.int32)

    
    test_ill_ = ills[int(len(ills) // 1 * args.rate):]
    test_ill = np.array(test_ill_, dtype=np.int32)

    test_left = torch.LongTensor(test_ill[:, 0].squeeze()).to(device)
    test_right = torch.LongTensor(test_ill[:, 1].squeeze()).to(device)

    left_non_train = list(set(left_ents) - set(train_ill[:, 0].tolist()))
    right_non_train = list(set(right_ents) - set(train_ill[:, 1].tolist()))
    #left_non_train = test_ill[:,0].tolist()
    #right_non_train = test_ill[:,1].tolist()
    print ("#left entity : %d, #right entity: %d" % (len(left_ents), len(right_ents)))
    print ("#left entity not in train set: %d, #right entity not in train set: %d" 
            % (len(left_non_train), len(right_non_train)))
    
    rel_features = load_relation(ENT_NUM, triples, 1000)
    rel_features = torch.Tensor(rel_features).to(device)
    print ("relation feature shape:", rel_features.shape)
    
    a1 = os.path.join(args.file_dir, 'training_attrs_1')
    a2 = os.path.join(args.file_dir, 'training_attrs_2')
    att_features = load_attr([a1, a2], ENT_NUM, ent2id_dict, 1000) # attr
    att_features = torch.Tensor(att_features).to(device)
    print ("attribute feature shape:", att_features.shape)
    
    rel_fc = nn.Linear(1000, 100).to(device)
    att_fc = nn.Linear(1000, 100).to(device)
    img_fc = nn.Linear(img_features.shape[1], 200).to(device)

    print("-----dataset summary-----")
    print("dataset:\t", args.file_dir)
    print("triple num:\t", len(triples))
    print("entity num:\t", ENT_NUM)
    print("relation num:\t", REL_NUM)
    print("train ill num:\t", train_ill.shape[0], "\ttest ill num:\t", test_ill.shape[0])
    print("-------------------------")

    input_dim = int(args.hidden_units.strip().split(",")[0])

    entity_emb = nn.Embedding(ENT_NUM, input_dim)
    nn.init.normal_(entity_emb.weight, std=1.0 / math.sqrt(ENT_NUM))
    entity_emb.requires_grad = True
    entity_emb = entity_emb.to(device)

    input_idx = torch.LongTensor(np.arange(ENT_NUM)).to(device)
    
    adj = get_adjr(ENT_NUM, triples, norm=True)
    adj = adj.to(device)

    # Set model
    n_units = [int(x) for x in args.hidden_units.strip().split(",")]
    n_heads = [int(x) for x in args.heads.strip().split(",")]
    
    # GAT
    #cross_graph_model = GAT(n_units=n_units, n_heads=n_heads, dropout=args.dropout, attn_dropout=args.attn_dropout,\
    #  instance_normalization=args.instance_normalization, diag=True).to(device)
    # std GCN
    cross_graph_model = GCN(n_units[0], n_units[1], n_units[2], dropout=args.dropout).to(device) 

    for param in img_fc.parameters():
        param.requires_grad = True

    params = [
            {"params":
            list(cross_graph_model.parameters())+
            list(img_fc.parameters()) + 
            list(rel_fc.parameters()) + 
            list(att_fc.parameters()) +
            [entity_emb.weight] + 
            [weight_raw]
            }]
    optimizer = optim.AdamW(
            params,
            lr=args.lr
    )
    #{"params": [weight_raw], "lr":0.01, "weight_decay":0}], 
    #optimizer = optim.AdamW(params, lr=args.lr)
    print ("GCN model details:")
    print(cross_graph_model)
    print ("optimiser details:")
    print(optimizer)

    # modality-specific loss
    criterion_gcn = NCA_loss(alpha=5, beta=10, ep=0.0)
    criterion_rel = NCA_loss(alpha=15, beta=10, ep=0.0)
    criterion_att = NCA_loss(alpha=15, beta=10, ep=0.0)
    criterion_img = NCA_loss(alpha=15, beta=10, ep=0.0)
    
    # joint loss
    criterion_all = NCA_loss(alpha=15, beta=10, ep=0.0)
    
    # Train
    print("[start training...] ")
    t_total = time.time()
    new_links = []
    epoch_KE, epoch_CG = 0, 0
    for epoch in range(args.epochs):

        if epoch == epoch >= args.il_start: 
            optimizer = optim.AdamW(params, lr=args.lr / 5 )
        
        t_epoch = time.time()
        cross_graph_model.train()
        img_fc.train()
        rel_fc.train()
        att_fc.train()
        optimizer.zero_grad()
        
        gph_emb = cross_graph_model(entity_emb(input_idx), adj)
        img_emb = img_fc(img_features)
        rel_emb = rel_fc(rel_features)
        att_emb = att_fc(att_features)
        loss_sum_gcn, loss_sum_rel, loss_sum_att, loss_sum_img, loss_sum_all = 0,0,0,0,0

        epoch_CG += 1

        # manual batching
        np.random.shuffle(train_ill)
        for si in np.arange(0, train_ill.shape[0], args.bsize):
            
            w_normalized = F.softmax(weight_raw, dim=0)

            joint_emb = torch.cat([
                w_normalized[0] * F.normalize(img_emb).detach(), \
                w_normalized[1] * F.normalize(att_emb).detach(), \
                w_normalized[2] * F.normalize(rel_emb).detach(), \
                w_normalized[3] * F.normalize(gph_emb).detach(), \
                ], dim=1)
            #print(joint_emb.shape)
            #print (weight_raw)
            #print (left_emb.shape, right_emb.shape)
            loss_GCN = criterion_gcn(gph_emb, train_ill[si:si+args.bsize], [], device=device)
            loss_rel = criterion_rel(rel_emb, train_ill[si:si+args.bsize], [], device=device)
            loss_att = criterion_att(att_emb, train_ill[si:si+args.bsize], [], device=device)
            loss_img = criterion_img(img_emb, train_ill[si:si+args.bsize], [], device=device)
            loss_joi = criterion_img(joint_emb, train_ill[si:si+args.bsize], [], device=device)
            
            loss_all = loss_joi  + loss_att + loss_rel + loss_GCN + loss_img

            loss_all.backward(retain_graph=True)

            loss_sum_all = loss_sum_all + loss_all.item() 

        optimizer.step()
        #print("loss_GCN in epoch {:d}: {:f}, time: {:.4f} s".format(epoch, loss_sum_gcn, time.time() - t_epoch))
        #print("loss_rel in epoch {:d}: {:f}, time: {:.4f} s".format(epoch, loss_sum_rel, time.time() - t_epoch))
        #print("loss_att in epoch {:d}: {:f}, time: {:.4f} s".format(epoch, loss_sum_att, time.time() - t_epoch))
        #print("loss_img in epoch {:d}: {:f}, time: {:.4f} s".format(epoch, loss_sum_img, time.time() - t_epoch))
        print("[epoch {:d}] loss_all: {:f}, time: {:.4f} s".format(epoch, loss_sum_all, time.time() - t_epoch))

        del joint_emb, gph_emb, img_emb, rel_emb, att_emb

        # print weights to a csv file (for analysis)
        #with open("logs/weights_4_3.csv", "a") as myfile:
        #    myfile.write("%.5f\t%.5f\t%.5f\t%.5f\n" % (w_normalized[0].item(),w_normalized[1].item(),\
        #        w_normalized[2].item(),w_normalized[3].item()))

        
        # semi-supervised learning
        if epoch >= args.il_start and (epoch+1) % args.semi_learn_step == 0 and args.il:
            # predict links
            with torch.no_grad():
                w_normalized = F.softmax(weight_raw, dim=0)

                final_emb = torch.cat([
                    w_normalized[0] * F.normalize(img_fc(img_features)), \
                    w_normalized[1] * F.normalize(att_fc(att_features)), \
                    w_normalized[2] * F.normalize(rel_fc(rel_features)), \
                    w_normalized[3] * F.normalize(cross_graph_model(entity_emb(input_idx), adj))
                ], dim=1)
                    
                final_emb = F.normalize(final_emb)
            
            distance_list = []
            for i in np.arange(0,len(left_non_train), 1000):
                d = pairwise_distances(final_emb[left_non_train[i:i+1000]], final_emb[right_non_train])
                distance_list.append(d)
            distance = torch.cat(distance_list, dim=0)
            preds_l = torch.argmin(distance, dim=1).cpu().numpy().tolist()
            preds_r = torch.argmin(distance.t(), dim=1).cpu().numpy().tolist()
            del distance_list, distance, final_emb

            #if args.csls is True:
            #    distance = 1 - csls_sim(1 - distance, args.csls_k)
            #print (len(preds_l), len(preds_r))
        
            if (epoch+1) % (args.semi_learn_step * 10) == args.semi_learn_step:
                new_links = [(left_non_train[i],right_non_train[p]) for i,p in enumerate(preds_l) if preds_r[p]==i]
            else:
                new_links = [(left_non_train[i],right_non_train[p]) for i,p in enumerate(preds_l) if (preds_r[p]==i) \
                    and ((left_non_train[i],right_non_train[p]) in new_links)]
            print ("[epoch %d] #links in candidate set: %d" %(epoch, len(new_links)))
        
        if epoch >= args.il_start and (epoch+1) % (args.semi_learn_step * 10) == 0 and len(new_links)!=0 and args.il:
            # get similarity of newly linked pairs
            #if len(new_links) > 1000:
            #    left_inds = np.array(new_links, dtype=np.int32)[:,0] 
            #    right_inds = np.array(new_links, dtype=np.int32)[:,1]
            #    emb_sim = final_emb[left_inds].mm(final_emb[right_inds].t())
            #    two_d_indices = get_topk_indices(emb_sim, K=1000)
            #    new_links_elect = []
            #    for inds in two_d_indices:
            #        new_links_elect.append((left_inds[inds[0]], right_inds[inds[1]]))
            #else:
            new_links_elect = new_links
            print ("\n#new_links_elect:", len(new_links_elect))

            #if len(new_links) >= 5000: new_links = random.sample(new_links, 5000)
            train_ill = np.vstack((train_ill, np.array(new_links_elect)))
            print ("train_ill.shape:", train_ill.shape)

            num_true = len([nl for nl in new_links_elect if nl in test_ill_])
            print ("#true_links: %d" % num_true)
            print ("true link ratio: %.1f%%" % (100 * num_true / len(new_links_elect)))

            # remove from left/right_non_train
            for nl in new_links_elect:
                left_non_train.remove(nl[0])
                right_non_train.remove(nl[1])
            print ("#entity not in train set: %d (left) %d (right)" % (len(left_non_train), len(right_non_train)))

            new_links = []
        
        if args.cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Test
        if (epoch + 1) % args.check_point == 0:
            print("\n[epoch {:d}] checkpoint!".format(epoch))
            
            with torch.no_grad():
                t_test = time.time()
                cross_graph_model.eval()
                img_fc.eval()
                rel_fc.eval()
                att_fc.eval()

                gph_emb = cross_graph_model(entity_emb(input_idx), adj)
                img_emb = img_fc(img_features)
                rel_emb = rel_fc(rel_features)
                att_emb = att_fc(att_features)

                w_normalized = F.softmax(weight_raw, dim=0)
                print ("normalised weights:", w_normalized)

                final_emb = torch.cat([
                    w_normalized[0] * F.normalize(img_emb), \
                    w_normalized[1] * F.normalize(att_emb), \
                    w_normalized[2] * F.normalize(rel_emb), \
                    w_normalized[3] * F.normalize(gph_emb), \
                    ], dim=1)

                final_emb = F.normalize(final_emb)

                #top_k = [1, 5, 10, 50, 100]
                top_k = [1, 10, 50]
                if "100" in args.file_dir:
                    Lvec = final_emb[test_left].cpu().data.numpy()
                    Rvec = final_emb[test_right].cpu().data.numpy()
                    acc_l2r, mean_l2r, mrr_l2r, acc_r2l, mean_r2l, mrr_r2l = multi_get_hits(Lvec, Rvec, top_k=top_k, args=args)
                    del final_emb
                    gc.collect()
                else:
                    acc_l2r = np.zeros((len(top_k)), dtype=np.float32)
                    acc_r2l = np.zeros((len(top_k)), dtype=np.float32)
                    test_total, test_loss, mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0, 0., 0., 0., 0., 0.
                    if args.dist == 2:
                        distance = pairwise_distances(final_emb[test_left], final_emb[test_right])
                    elif args.dist == 1:
                        distance = torch.FloatTensor(scipy.spatial.distance.cdist(\
                            final_emb[test_left].cpu().data.numpy(),\
                            final_emb[test_right].cpu().data.numpy(), metric="cityblock"))
                    else:
                        raise NotImplementedError
                    
                    if args.csls is True:
                        distance = 1 - csls_sim(1 - distance, args.csls_k)
                    
                    if epoch+1 == args.epochs:
                        to_write = []
                        test_left_np = test_left.cpu().numpy()
                        test_right_np = test_right.cpu().numpy()
                        to_write.append(["idx","rank", "query_id", "gt_id", "ret1", "ret2", "ret3"])
                    for idx in range(test_left.shape[0]):
                        values, indices = torch.sort(distance[idx, :], descending=False)
                        rank = (indices == idx).nonzero().squeeze().item()
                        mean_l2r += (rank + 1)
                        mrr_l2r += 1.0 / (rank + 1)
                        for i in range(len(top_k)):
                            if rank < top_k[i]:
                                acc_l2r[i] += 1
                        # save idx, correct rank pos, and indices
                        if epoch+1 == args.epochs:
                            indices = indices.cpu().numpy()
                            to_write.append([idx, rank, test_left_np[idx], test_right_np[idx], test_right_np[indices[0]], test_right_np[indices[1]], test_right_np[indices[2]]])
                    if epoch+1 == args.epochs:
                        import csv
                        with open("logs/pred.txt", "w") as f:
                            wr = csv.writer(f, dialect='excel')
                            wr.writerows(to_write)

                    for idx in range(test_right.shape[0]):
                        _, indices = torch.sort(distance[:, idx], descending=False)
                        rank = (indices == idx).nonzero().squeeze().item()
                        mean_r2l += (rank + 1)
                        mrr_r2l += 1.0 / (rank + 1)
                        for i in range(len(top_k)):
                            if rank < top_k[i]:
                                acc_r2l[i] += 1
                    mean_l2r /= test_left.size(0)
                    mean_r2l /= test_right.size(0)
                    mrr_l2r /= test_left.size(0)
                    mrr_r2l /= test_right.size(0)
                    for i in range(len(top_k)):
                        acc_l2r[i] = round(acc_l2r[i] / test_left.size(0), 4)
                        acc_r2l[i] = round(acc_r2l[i] / test_right.size(0), 4)
                    del distance, gph_emb, img_emb, rel_emb, att_emb
                    gc.collect()
                print("l2r: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s ".format(top_k, acc_l2r, mean_l2r, mrr_l2r, time.time() - t_test))
                print("r2l: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s \n".format(top_k, acc_r2l, mean_r2l, mrr_r2l, time.time() - t_test))

        if args.cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("[optimization finished!]")
    print("[total time elapsed: {:.4f} s]".format(time.time() - t_total))

if __name__ == "__main__":
    main()
