import pandas as pd
import pyarrow.parquet as pq
from fastparquet import ParquetFile
from top2vec import Top2Vec
import hnswlib

import umap
import umap.utils as utils
import umap.aligned_umap
import umap.plot

import hdbscan
from sklearn.cluster import DBSCAN

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.rcParams.update({'figure.max_open_warning': 0})
import itertools
from math import dist

import seaborn as sns
from celluloid import Camera

import pickle

from scipy import spatial
from scipy.spatial import distance

import collections



def clean_df(df, start, end) :
    """
    fonction qui enleve ce qu'on ne veut pas du dataframe
    """
    df_cleaned = df.dropna(subset=['abstract']) #enleve None de abstract
    df_cleaned = df_cleaned.drop(df_cleaned[df_cleaned.abstract == ""].index) #enleve vide de abstract
    df_cleaned = df_cleaned.dropna(subset=['year']) #enleve None de year
    df_cleaned = df_cleaned.drop(df_cleaned[df_cleaned.year == 0].index) #enleve 0 de year
    
    df_cleaned = df_cleaned.sort_values(["year","_id"]) #sort by year & _id
    df_cleaned = df_cleaned[df_cleaned['year'].between(start, end)] # entre start et end
    
    df_cleaned = df_cleaned.reset_index()
    return df_cleaned

def slice_by_year(list_zip, min_year, max_year, nb_years, overlap_years) :
    """
    list(year,id,doc_vectors) -> list of list of doc_vectors, list of list of ids
    we've slice in parts of nb_years and with an overlap between slices of overlap_years
    it gives the ids slices in periods (for the second argument of AlignedUMAP) and the first argument of AlignedUMAP
    """
    if(nb_years < overlap_years) : 
        return []
    
    sliced_list_id = []
    sliced_list_dv = []
    list_tmp_id = []
    list_tmp_dv = []
    curr_year = min_year
    
    while (curr_year < max_year) : 
        period = [curr_year + i for i in range(0,nb_years)]
        for i in range(len(list_zip)) : 
            if list_zip[i][0] in period : 
                list_tmp_id.append(list_zip[i][1])
                list_tmp_dv.append(list_zip[i][2])
        sliced_list_id.append(list_tmp_id)
        sliced_list_dv.append(list_tmp_dv)
        list_tmp_id = []
        list_tmp_dv = []
        curr_year = curr_year + nb_years - overlap_years
        
    return sliced_list_dv, sliced_list_id

def relation_periodes(list_ids1, list_ids2) :
    """
    Donne le dictionnaire de relation de documments communs entre deux periodes
    sert pour deuxieme arguement de AlignedUMAP
    """
    D = dict()
    for i in list_ids1 : 
        if i in list_ids2 : 
            D[list_ids1.index(i)] = list_ids2.index(i)
    return D

def relations_periodes(list_list_id) :
    """
    Donne la liste de dictionnaires des relationsde documments communs entre deux periodes
    sert pour deuxieme arguement de AlignedUMAP
    """
    L = []
    for i in range(len(list_list_id)-1) :
        L.append(relation_periodes(list_list_id[i],list_list_id[i + 1]))
    return L


"""
enregistrement de resultat de AlignedUMAP

with open("embedding", "wb") as fp:
    pickle.dump(list(embedding), fp)
    
with open("embedding", "rb") as fp:
    embedding = pickle.load(fp)
"""

def hdbscan_plot(embedding, min_cluster_size_nb, min_samples_nb) :
    """
    applique HDBSCAN au resultat de alignedUMAP
    il affiche les clusters a l'ecran
    on obtient une liste de clusters, un groupe de clusters par periode
    """
    clusters = []
    for e in range(len(embedding)) :
        c = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size_nb, min_samples = min_samples_nb, metric = "euclidean",cluster_selection_method = "eom").fit(embedding[e])
        clusters.append(c)
        color_palette = sns.color_palette(n_colors = c.labels_.max()+1)
        cluster_colors = [color_palette[x] if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in c.labels_]
        cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                 zip(cluster_colors, c.probabilities_)]
        plt.scatter(*embedding[e].T, s=5, c=cluster_member_colors, alpha=1)
        plt.show()
    return clusters

def dbscan_plot(embedding, eps, min_samples) :
    """
    applique DBSCAN au resultat de alignedUMAP
    il affiche les clusters a l'ecran
    on obtient une liste de clusters, un groupe de clusters par periode
    """
    clusters = []
    for e in range(len(embedding)) :
        db = DBSCAN(eps = eps, min_samples = min_samples).fit(embedding[e])
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        clusters.append(db)
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = labels == k

            xy = embedding[e][class_member_mask & core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor=tuple(col),
                markersize=4,
            )

            xy = embedding[e][class_member_mask & ~core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor=tuple(col),
                markersize=1,
            )

        plt.title("Estimated number of clusters: %d" % n_clusters_)
        plt.show()

    return clusters

def relations_clusters_doc_cummuns(clusters, relations_periodes) : 
    """
    fonction qui donne tous les association des clusters en s'appuyant sur les documents communs
    """
    list_rels = []
    for i in range(len(clusters)-1) :
        rel0 = [clusters[i].labels_[j] for j in relations_periodes[i].keys()]
        rel1 = [clusters[i+1].labels_[j] for j in relations_periodes[i].values()]
        cluster_relations = zip(rel0,rel1)
        l_cluster_relations = [list(x) for x in cluster_relations] 
        l = [i for i in l_cluster_relations if i[0] != -1 and i[1] != -1]
        list_rel = []
        # calculer fréquence de chaque obj dans rel_p0p1
        list_rel = []
        while l != [] :
            list_rel.append((l[0],l.count(l[0])))
            l = list(filter((l[0]).__ne__, l))
        list_rels.append(list_rel)
    return list_rels

def relation_cluster_tous_doc(cluster_p0, cluster_p1, embedding_p0, embedding_p1):
    """
    fonction qui donne l'association de clusters entre 2 périodes en prenant en compte tous les documents
    """
    clusterLabel_vector2D_p0 = list(zip(cluster_p0.labels_, embedding_p0))
    clusterLabel_vector2D_p1 = list(zip(cluster_p1.labels_, embedding_p1))
    rel_p0p1 = []
    for i in clusterLabel_vector2D_p0 :
        c0, v0 = i
        if c0 == -1 :
            rel_p0p1.append([-1,-1])
        else :
            dist,ind = spatial.KDTree(embedding_p1).query(v0)
            c1, v1 = clusterLabel_vector2D_p1[ind]
            rel_p0p1.append([c0,c1])
    # enlever les relations pas interessants, ceux avec -1 
    rel_p0p1 = [i for i in rel_p0p1 if i[0] != -1 and i[1] != -1]
    # calculer fréquence de chaque obj dans rel_p0p1
    rel = []
    while rel_p0p1 != [] :
        rel.append((rel_p0p1[0],rel_p0p1.count(rel_p0p1[0])))
        rel_p0p1 = list(filter((rel_p0p1[0]).__ne__, rel_p0p1))
    return rel

def tous_relations_cluster_tous_doc(clusters, embedding) :
    """
    fonction qui donne tous les relations de clusters en prenant en compte tous les documents
    """
    list_rel = []
    for i in range(len(embedding)-1) :
        list_rel.append(relation_cluster_tous_doc(clusters[i], clusters[i+1], embedding[i], embedding[i+1])) 
    return list_rel

def relation_cluster_tous_doc_HD(cluster_p0, cluster_p1, list_dv_p0, list_dv_p1):
    """
    fonction qui donne l'association de clusters entre 2 périodes en prenant en compte tous les documents en haute dimension
    """
    clusterLabel_vectorHD_p0 = list(zip(cluster_p0.labels_, list_dv_p0))
    clusterLabel_vectorHD_p1 = list(zip(cluster_p1.labels_, list_dv_p1))
    rel_p0p1 = []
    for i in clusterLabel_vectorHD_p0 :
        c0, v0 = i
        if c0 == -1 :
            rel_p0p1.append([-1,-1])
        else :
            dist,ind = spatial.KDTree(list_dv_p1).query(v0)
            c1, v1 = clusterLabel_vectorHD_p1[ind]
            rel_p0p1.append([c0,c1])
    # enlever les relations pas interessants, ceux avec -1 
    rel_p0p1 = [i for i in rel_p0p1 if i[0] != -1 and i[1] != -1]
    # calculer fréquence de chaque obj dans rel_p0p1
    rel = []
    while rel_p0p1 != [] :
        rel.append((rel_p0p1[0],rel_p0p1.count(rel_p0p1[0])))
        rel_p0p1 = list(filter((rel_p0p1[0]).__ne__, rel_p0p1))
    return rel

def tous_relations_cluster_tous_doc_HD(clusters, list_dv) :
    """
    fonction qui donne tous les relations de clusters en prenant en compte tous les documents en haute dimension
    """
    list_rel = []
    for i in range(len(list_dv)-1) :
        list_rel.append(relation_cluster_tous_doc(clusters[i], clusters[i+1], list_dv[i], list_dv[i+1])) 
    return list_rel

def plot_rel_clusters (embedding, clusters, list_rel) :
    """fonction qui plot tous les relations de clusters"""
    for i in range(len(embedding)-1) : 
        clusterLabel_vector2D_p0 = list(zip(clusters[i].labels_, embedding[i]))
        clusterLabel_vector2D_p1 = list(zip(clusters[i+1].labels_, embedding[i+1]))
        for l,f in list_rel[i]: 
            list_v2D_cluster0 = [v for c,v in clusterLabel_vector2D_p0 if c==l[0]]
            list_v2D_cluster1 = [v for c,v in clusterLabel_vector2D_p1 if c==l[1]]
            print("period " + str(i) + " & period " + str(i+1))
            print("Pour l'association de clusters " + str(l) + " qui a " + str(f) + " occurences")
            plt.scatter(*np.array(list_v2D_cluster0).T, s=2, c='b', alpha=1)
            plt.scatter(*np.array(list_v2D_cluster1).T, s=2, c='r', alpha=1)
            plt.show()
            
def plot_rel_clusters_link (embedding, clusters, list_rel, start, xmin=-20, xmax=20, ymin=-20, ymax=20) :
    """fonction qui plot l'evolution du cluster start au cours du temps"""
    period_cluster = [(0,start)]
    print("A partir de periode " + str(period_cluster[0][0]) + " et cluster " + str(period_cluster[0][1]))
    tmp = set()
    color_cycle= itertools.cycle(sns.color_palette("hsv",20))
    #color = "bgrcmyk"*len(list_rel)*2
    legend = []
    cluster_link = []
    while period_cluster != [] : 
        if period_cluster[0][0] != len(list_rel) :
            clusterLabel_vector2D_p0 = list(zip(clusters[period_cluster[0][0]].labels_, embedding[period_cluster[0][0]]))
            for l,f in list_rel[period_cluster[0][0]]: 
                if l[0] == period_cluster[0][1] : 
                    if tmp == set() :
                        tmp.add(l[1])
                        list_v2D_cluster0 = [v for c,v in clusterLabel_vector2D_p0 if c==l[0]]
                        #plt.ylim(ymin,ymax)
                        #plt.xlim(xmin,xmax)
                        plt.scatter(*np.array(list_v2D_cluster0).T, s=2, color=next(color_cycle), alpha=1)
                        #color = color[1:]
                        cluster_link.append(period_cluster[0])
                    else : 
                        tmp.add(l[1])
            legend.append(period_cluster[0])
            for elem in tmp :
                period_cluster.append((period_cluster[0][0]+1,elem))
            #plt.show()
            
            period_cluster.pop(0)
            tmp = set()
        else : 
            period_cluster = list(set(period_cluster))
            clusterLabel_vector2D_p0 = list(zip(clusters[period_cluster[0][0]].labels_, embedding[period_cluster[0][0]]))
            list_v2D_cluster0 = [v for c,v in clusterLabel_vector2D_p0 if c==period_cluster[0][1]]
            #plt.ylim(ymin,ymax)
            #plt.xlim(xmin,xmax)
            plt.scatter(*np.array(list_v2D_cluster0).T, s=2, color=next(color_cycle), alpha=1)
            #color = color[1:]
            legend.append(period_cluster[0])
            cluster_link.append(period_cluster[0])
            period_cluster.pop(0)
    plt.legend(legend)
    plt.show()
    return cluster_link


def plot_rel_clusters_link_apart (embedding, clusters, list_rel, start, xmin=-10, xmax=10, ymin=-10, ymax=10) :
    """fonction qui plot l'evolution du cluster start au cours du temps"""
    period_cluster = [(0,start)]
    print("A partir de periode " + str(period_cluster[0][0]) + " et cluster " + str(period_cluster[0][1]))
    tmp = set()
    color_cycle= itertools.cycle(sns.color_palette("hsv",20))
    cluster_link = []
    while period_cluster != [] : 
        if period_cluster[0][0] != len(list_rel) :
            clusterLabel_vector2D_p0 = list(zip(clusters[period_cluster[0][0]].labels_, embedding[period_cluster[0][0]]))
            for l,f in list_rel[period_cluster[0][0]]: 
                if l[0] == period_cluster[0][1] : 
                    if tmp == set() :
                        tmp.add(l[1])
                        list_v2D_cluster0 = [v for c,v in clusterLabel_vector2D_p0 if c==l[0]]
                        plt.ylim(ymin,ymax)
                        plt.xlim(xmin,xmax)
                        plt.scatter(*np.array(list_v2D_cluster0).T, s=2, color=next(color_cycle), alpha=1)
                        plt.show()
                        cluster_link.append(period_cluster[0])
                        print(period_cluster[0])

                    else : 
                        tmp.add(l[1])

            for elem in tmp :
                period_cluster.append((period_cluster[0][0]+1,elem))
            period_cluster.pop(0)
            tmp = set()
        else : 
            period_cluster = list(set(period_cluster))
            clusterLabel_vector2D_p0 = list(zip(clusters[period_cluster[0][0]].labels_, embedding[period_cluster[0][0]]))
            list_v2D_cluster0 = [v for c,v in clusterLabel_vector2D_p0 if c==period_cluster[0][1]]
            plt.ylim(ymin,ymax)
            plt.xlim(xmin,xmax)
            plt.scatter(*np.array(list_v2D_cluster0).T, s=2, color=next(color_cycle), alpha=1)
            plt.show()
            print(period_cluster[0])
            cluster_link.append(period_cluster[0])
            period_cluster.pop(0)
    return cluster_link
    
def create_new_labels_with_list_rels(df, list_rels, clusters, sliced_list_id) :
    """
    fonction auxiliaire pour creer la liste de labels, parametre pour _create_topic_vectors selon une liste de relations de clusters
    """
    list_rels_label = []
    for ind in range(len(list_rels)) : 
        #indice de décalage
        ind_debut_cluster0 = df._id.tolist().index(sliced_list_id[ind][0])
        ind_debut_cluster1 = df._id.tolist().index(sliced_list_id[ind+1][0])
        print(ind_debut_cluster0)
        print(ind_debut_cluster1)

        rels_label = []
        for rel in list_rels[ind] : 
            fst = rel[0]
            l_ind_cluster0 = [i for i in range(len(clusters[ind].labels_)) if clusters[ind].labels_[i] == fst]
            l_ind_cluster0_v2 = [i + ind_debut_cluster0 for i in l_ind_cluster0] 

            snd = rel[1] 
            l_ind_cluster1 = [i for i in range(len(clusters[ind + 1].labels_)) if clusters[ind + 1].labels_[i] == snd]
            l_ind_cluster1_v2 = [i + ind_debut_cluster1 for i in l_ind_cluster1] 

            rel_label = []
            new_labels = []

            for i in range(len(df)) :
                if i not in l_ind_cluster0_v2 : 
                    new_labels.append(-1)
                else : 
                    new_labels.append(clusters[ind].labels_[i - ind_debut_cluster0])
            rel_label.append(new_labels)   
            new_labels = []
            for i in range(len(df)) :
                if i not in l_ind_cluster1_v2 :   
                    new_labels.append(-1)
                else : 
                    new_labels.append(clusters[ind+1].labels_[i - ind_debut_cluster1])
            rel_label.append(new_labels)   
            rels_label.append(rel_label)
        list_rels_label.append(rels_label)
    return list_rels_label

def create_new_labels_for_all_clusters(df, clusters, sliced_list_id) :
    """
    fonction auxiliaire pour creer la liste de labels, parametre pour _create_topic_vectors pour tous les clusters en ordre de periode et numero de cluster
    """
    list_labels = []
    for ind in range(len(clusters)) : 
        #indice de décalage
        ind_debut_cluster0 = df._id.tolist().index(sliced_list_id[ind][0])
        list_label = []
        for j in range(clusters[ind].labels_.max()+1) : 
            l_ind_cluster0 = [i for i in range(len(clusters[ind].labels_)) if clusters[ind].labels_[i] == j]
            l_ind_cluster0_v2 = [i + ind_debut_cluster0 for i in l_ind_cluster0] 
            new_labels = []

            for i in range(len(df)) :
                if i not in l_ind_cluster0_v2 : 
                    new_labels.append(-1)
                else : 
                    new_labels.append(clusters[ind].labels_[i - ind_debut_cluster0])
            list_label.append(new_labels)
        list_labels.append(list_label)
    return list_labels

def rel_to_topic_words (model, list_rel, list_rels_label) :
    """
    fonction qui donne a partir d'une liste de labels, les topic_words et les topic_word_scores en prenant en compte l'ordre de liste de relations
    """
    rels_topic_words = []
    for i in range(len(list_rel)): 
        rel_topic_words = []
        fst_rel, snd_rel = list_rel[i] 
        fst_label, snd_label = list_rels_label[i]
        
        model._create_topic_vectors(fst_label)
        model._deduplicate_topics()
        model.topic_words, model.topic_word_scores = model._find_topic_words_and_scores(topic_vectors=model.topic_vectors)
        rel_topic_words.append([model.topic_words.copy(),model.topic_word_scores.copy()])

        model._create_topic_vectors(snd_label)
        model._deduplicate_topics()
        model.topic_words, model.topic_word_scores = model._find_topic_words_and_scores(topic_vectors=model.topic_vectors)
        rel_topic_words.append([model.topic_words.copy(),model.topic_word_scores.copy()])
        
        rels_topic_words.append(rel_topic_words)
    return rels_topic_words


def labels_to_topic_words (model, list_labels) :
    """
    fonction qui donne a partir d'une liste de labels, les topic_words et les topic_word_scores
    """
    list_topic_words = []
    for i in range(len(list_labels)): 
        topic_words_periode = []
        for j in range(len(list_labels[i])):
            model._create_topic_vectors(list_labels[i][j])
            model._deduplicate_topics()
            model.topic_words, model.topic_word_scores = model._find_topic_words_and_scores(topic_vectors=model.topic_vectors)
            topic_words_periode.append([model.topic_words.copy(),model.topic_word_scores.copy()])
            #model.generate_topic_wordcloud(0)
            
        list_topic_words.append(topic_words_periode)
    return list_topic_words

def rel_to_topic_vectors(model, list_rel, list_rels_label) :
    """
    fonction qui donne a partir d'une liste de labels, les topic_vectors en prenant en compte l'ordre de liste de relations
    """
    rels_topic_vectors = []
    for i in range(len(list_rel)): 
        rel_topic_vectors = []
        fst_rel, snd_rel = list_rel[i] 
        fst_label, snd_label = list_rels_label[i]

        model._create_topic_vectors(fst_label)
        rel_topic_vectors.append(model.topic_vectors)
        #model.generate_topic_wordcloud(0)

        model._create_topic_vectors(snd_label)
        rel_topic_vectors.append(model.topic_vectors)
        #model.generate_topic_wordcloud(0)
        
        rels_topic_vectors.append(rel_topic_vectors)
    return rels_topic_vectors

def all_rel_to_topic_vectors(model, list_rels, list_rels_label) :
    all_rels_topic_vectors = []
    for i in range(len(list_rels)) :
        all_rels_topic_vectors.append(rel_to_topic_vectors(model, list_rels[i], list_rels_label[i]))
    return all_rels_topic_vectors

def rel_to_topic_vectors_words(model, list_rel, list_rels_label) :
    """
    fonction qui donne a partir d'une liste de labels, les topic_vectors en prenant en compte l'ordre de liste de relations
    """
    rels_topic_vectors = []
    rels_topic_words = []
    for i in range(len(list_rel)): 
        rel_topic_vectors = []
        rel_topic_words = []
        fst_rel, snd_rel = list_rel[i] 
        fst_label, snd_label = list_rels_label[i]

        model._create_topic_vectors(fst_label)
        rel_topic_vectors.append(model.topic_vectors)
        model.topic_words, model.topic_word_scores = model._find_topic_words_and_scores(topic_vectors=model.topic_vectors)
        rel_topic_words.append(model.topic_words)
        #model.generate_topic_wordcloud(0)

        model._create_topic_vectors(snd_label)
        rel_topic_vectors.append(model.topic_vectors)
        model.topic_words, model.topic_word_scores = model._find_topic_words_and_scores(topic_vectors=model.topic_vectors)
        rel_topic_words.append(model.topic_words)
        #model.generate_topic_wordcloud(0)
        rels_topic_vectors.append(rel_topic_vectors)
        rels_topic_words.append(rel_topic_words)
    return rels_topic_vectors, rels_topic_words

def all_rel_to_topic_vectors_words(model, list_rels, list_rels_label) :
    all_rels_topic_vectors = []
    all_rels_topic_words = []
    for i in range(len(list_rels)) :
        all_rels_topic_vectors.append(rel_to_topic_vectors_words(model, list_rels[i], list_rels_label[i])[0])
        all_rels_topic_words.append(rel_to_topic_vectors_words(model, list_rels[i], list_rels_label[i])[1])
    return all_rels_topic_vectors, all_rels_topic_words

def topic_vectors_link (cluster_link, all_topic_vectors) :
    """fonction qui donne l'evolution de topics au cours du temps """
    l_topic_vectors_link = []
    
    for p,c in cluster_link : 
        if p < len(all_topic_vectors):
            l_topic_vectors_link.append(all_topic_vectors[p][c][0][0])
        else : 
            l_topic_vectors_link.append(all_topic_vectors[p-1][c][1][0])

    return l_topic_vectors_link


def same_words_2conseq_periods_diff_score (model, list_rels, list_rels_label, between_period_ind,fst_cluster, snd_cluster) : 
    """
    fonction qui retourne les mots communs et leur difference de score entre deux clusters de deux periodes consecutives
    """
    list_topic_words = rel_to_topic_words(model, list_rels[between_period_ind], list_rels_label[between_period_ind])
    if [fst_cluster,snd_cluster] in list_rels[between_period_ind] : 
        ind = list_rels[between_period_ind].index([fst_cluster,snd_cluster])
    else : 
        print("error : [" + str(fst_cluster) + ", " + str(snd_cluster) + "] n'appartient pas a relation de clusters")
        return []
    words_period0_cluster1 = list(zip(list_topic_words[ind][0][0][0],list_topic_words[ind][0][1][0]))
    words_period0_cluster2 = list(zip(list_topic_words[ind][1][0][0],list_topic_words[ind][1][1][0]))
    same_words = []
    for words0 in words_period0_cluster1 : 
        w0,s0 = words0
        for words1 in words_period0_cluster2 : 
            w1,s1 = words1
            if w0 == w1 : 
                same_words.append((w0,s0 - s1)) # positif ca devient important, négatif ca devient moins important
    return same_words

def same_words_2conseq_periods_list_score (model, list_rels, list_rels_label, dict_same_words, between_period_ind,fst_cluster, snd_cluster) : 
    """
    fonction qui retourne les mots communs et leur score entre deux clusters de deux periodes consecutives
    """    
    length = between_period_ind + 2
    list_topic_words = rel_to_topic_words(model, list_rels[between_period_ind], list_rels_label[between_period_ind])
    if [fst_cluster,snd_cluster] in list_rels[between_period_ind] : 
        ind = list_rels[between_period_ind].index([fst_cluster,snd_cluster])
    else : 
        print("error : [" + str(fst_cluster) + ", " + str(snd_cluster) + "] n'appartient pas a relation de clusters")
        return []
    words_period0_cluster1 = list(zip(list_topic_words[ind][0][0][0],list_topic_words[ind][0][1][0]))
    words_period0_cluster2 = list(zip(list_topic_words[ind][1][0][0],list_topic_words[ind][1][1][0]))
    for words0 in words_period0_cluster1 : 
        w0,s0 = words0
        bo = 0
        for words1 in words_period0_cluster2 : 
            w1,s1 = words1
            if w0 == w1 :
                bo = 1
                if w0 not in dict_same_words :
                    dict_same_words[w0] = [s0,s1]
                    if length >= 2 :
                        dict_same_words[w0] = [0] * (length - 2)
                        dict_same_words[w0].extend([s0,s1])
                else : 
                    dict_same_words[w0].extend([s1])
        if not bo :
            if w0 not in dict_same_words : 
                dict_same_words[w0] = [s0,0]
                if length >= 2 :
                    dict_same_words[w0] = [0] * (length - 2)
                    dict_same_words[w0].extend([s0,0])
            else : 
                dict_same_words[w0].extend([0])
    for i in dict_same_words :
        if(len(dict_same_words[i]) != length) :
            dict_same_words[i].extend([0])
    return dict_same_words

def same_words_all_list_score(model, list_rels_c, list_rels_label, cluster_start) :
    """
    fonction qui retourne les mots communs et leur score respectivement du cluster dont ils apartiennent a la periode correspondante
    """  
    dict_same_words = {}
    for i in range(len(list_rels_c)) :
        bo = 0
        for c0,c1 in list_rels_c[i] : 
            if cluster_start == c0 :
                bo = 1
                cluster_end = c1
        if bo == 0 : 
            print("cluster not in list_rels_c")
            return {}
        dict_same_words = same_words_2conseq_periods_list_score (model, list_rels_c, list_rels_label, dict_same_words, i, cluster_start, cluster_end)
        cluster_start = cluster_end
    return dict_same_words

def same_words_2conseq_periods_list_score_2 (model, list_rels, list_rels_label, dict_same_words, between_period_ind,fst_cluster, snd_cluster) : 
    """
    fonction qui retourne les mots communs et leur score entre deux clusters de deux periodes consecutives en prenant en compte que les mots du premier cluster
    """    
    list_topic_words = rel_to_topic_words(model, list_rels[between_period_ind], list_rels_label[between_period_ind])
    if [fst_cluster,snd_cluster] in list_rels[between_period_ind] : 
        ind = list_rels[between_period_ind].index([fst_cluster,snd_cluster])
    else : 
        print("error : [" + str(fst_cluster) + ", " + str(snd_cluster) + "] n'appartient pas a relation de clusters")
        return []
    words_period0_cluster1 = list(zip(list_topic_words[ind][0][0][0],list_topic_words[ind][0][1][0]))
    words_period0_cluster2 = list(zip(list_topic_words[ind][1][0][0],list_topic_words[ind][1][1][0]))
    for w0,s0 in words_period0_cluster1 : 
        bo = 0
        for w1,s1 in words_period0_cluster2 : 
            if w0 == w1 :
                bo = 1
                if w0 in dict_same_words :
                    dict_same_words[w0].extend([s1])
        if not bo :
            if w0 in dict_same_words : 
                dict_same_words[w0].extend([0])
    for i in dict_same_words :
        if(len(dict_same_words[i]) != between_period_ind + 2) :
            dict_same_words[i].extend([0])
    return dict_same_words

def same_words_all_list_score_2 (model, list_rels_c, list_rels_label, cluster_start) :
    """
    fonction qui retourne les mots communs et leur score respectivement du cluster dont ils apartiennent a la periode correspondante qui utilise la fonction same_words_2conseq_periods_list_score_2
    """  
    ind = -1
    list_topic_words = rel_to_topic_words(model, list_rels_c[0], list_rels_label[0])
    for c1,c2 in list_rels_c[0] :
        if c1 == cluster_start :
            ind = list_rels_c[0].index([c1,c2])
    if ind == -1 : 
        print("error : [" + str(cluster_start) + "] n'est pas un cluster valable")
        print("cluster not in list_rels_c")
        return {}
    words_period0_cluster = list(zip(list_topic_words[ind][0][0][0],list_topic_words[ind][0][1][0]))

    dict_same_words = {w0:[s0] for w0,s0 in words_period0_cluster}
    for i in range(len(list_rels_c)) :
        bo = 0
        for c0,c1 in list_rels_c[i] : 
            if cluster_start == c0 :
                bo = 1
                cluster_end = c1
        if bo == 0 : 
            print("cluster not in list_rels_c")
            return {}
        dict_same_words = same_words_2conseq_periods_list_score_2 (model, list_rels_c, list_rels_label, dict_same_words, i, cluster_start, cluster_end)
        cluster_start = cluster_end
    return dict_same_words


def clusterNperiod1_to_clusterMperiodn(clusterperiod1, list_rels) : 
    """
    fonction qui retourne les mots communs pour un cluster de la premiere periode et le cluster correspondant de la derniere periode
    """
    c0 = []
    c1 = []
    for i in range(len(list_rels[0])) :
        if list_rels[0][i][0] == clusterperiod1 : 
            c0.append(list_rels[0][i][1])
    for i in range(len(list_rels)-1) : 
        for cluster0 in c0 :
            for j in range(len(list_rels[i+1])) :
                if list_rels[i+1][j][0] == cluster0 :
                    c1.append(list_rels[i+1][j][1])
        c0 = c1
        c1 = []
    return c0

def same_words_1st_and_last_periods(list_rels, list_topic_words,cluster_start):
    """
    fonction qui retourne les mots communs et leur difference de score pour un cluster de la premiere periode et le cluster correspondant de la derniere periode
    """
    ind = cluster_start
    for i in clusterNperiod1_to_clusterMperiodn(ind, list_rels) : 
        words_period0_cluster1 = list(zip(list_topic_words[0][ind][0][0],list_topic_words[0][ind][1][0]))
        words_period0_cluster2 = list(zip(list_topic_words[len(list_topic_words)-1][i][0][0],list_topic_words[len(list_topic_words)-1][i][1][0]))
        same_words = []
        for words0 in words_period0_cluster1 : 
            w0,s0 = words0
            for words1 in words_period0_cluster2 : 
                w1,s1 = words1
                if w0 == w1 : 
                    same_words.append((w0,s0 - s1)) # positif ca devient important, négatif ca devient moins important
    return same_words

def plot_word_score (dict_score) : 
    for i in dict_score.items() : 
        word, score = i
        length = len(list(dict_score.values())[0])
        x = np.arange(length)
        fig, ax = plt.subplots()
        ax.plot(x, score)
        ax.set(xlim=(0, length - 1), xticks=np.arange(0, length - 1),
           ylim=(0, 1), yticks=np.arange(0,1))
        fig.suptitle(word)

def intracluster_distance(embedding, cluster) :
    ds = []
    for i in range(max(cluster.labels_)):
        c = [embedding[j] for j in range(len(cluster.labels_)) if cluster.labels_[j] == i]
        s = 0
        for x in c  :
            for y in c :
                if x[0]!=y[0] or x[1]!=y[1] : 
                    s = s + dist(x,y)
        d = 1 / (len(c) * (len(c) - 1)) * s
        ds.append(d)
    if len(ds)!=0 :
        return sum(ds) / len(ds)

def plus_proche_eps(embedding, intrad0) :
    intrad = 0
    length = len(embedding)
    min_samples = length//100
    eps = length
    while eps>1 :
        eps = eps/10
    print("min_samples =",min_samples," & eps =",eps)
    while(abs(intrad0-intrad) > 0.001):
        cluster1 = dbscan(eps,min_samples,embedding)
        intrad1 = intracluster_distance(embedding,cluster1)
        cluster2 = dbscan(eps+0.01,min_samples,embedding)
        intrad2 = intracluster_distance(embedding,cluster2)
        cluster3 = dbscan(eps-0.01,min_samples,embedding)
        intrad3 = intracluster_distance(embedding,cluster3)
        lst = [intrad1,intrad2,intrad3]
        intrad = lst[min(range(len(lst)), key = lambda i: abs(lst[i]-intrad0))]
        print("intrad1",intrad1,"intrad2",intrad2,"intrad3",intrad3, "intrad", intrad)
        if intrad == intrad2 : 
            eps = eps + 0.01
        if intrad == intrad3 : 
            eps = eps - 0.01
        if intrad == intrad1 : 
            return eps,intrad
        
"""        
model.model.wv["develop"] #obtenir vecteur  partir de mot 
#att a normalisation de vector 
"""