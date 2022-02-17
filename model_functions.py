import io, json
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix


def normalize(dataset):
    # compute mean rating for each user
    mean = dataset.groupby(by='id', as_index=False)['successful'].mean()
    norm_ratings = pd.merge(dataset, mean, suffixes=('','_mean'), on='id')
    
    # normalize each rating by substracting the mean rating of the corresponding user
    norm_ratings['norm_successful'] = norm_ratings['successful'] - norm_ratings['successful_mean']
    return mean.to_numpy()[:, 1], norm_ratings


def rating_matrix(dataframe, column):
    crosstab = pd.crosstab(dataframe.therapy, dataframe.id, dataframe[f'{column}'], aggfunc=sum).fillna(0).values
    matrix = csr_matrix(crosstab)
    return matrix


def create_model(rating_matrix, metric):
    """
    - create the nearest neighbors model with the corresponding similarity metric
    - fit the model
    """
    model = NearestNeighbors(metric=metric, n_neighbors=21, algorithm='brute')
    model.fit(rating_matrix)    
    return model

def nearest_neighbors(rating_matrix, model):
    """    
    :param rating_matrix : rating matrix of shape (nb_users, nb_items)
    :param model : nearest neighbors model    
    :return
        - similarities : distances of the neighbors from the referenced user
        - neighbors : neighbors of the referenced user in decreasing order of similarities
    """    
    similarities, neighbors = model.kneighbors(rating_matrix)        
    return similarities[:, 1:], neighbors[:, 1:]


def candidate_items(userid,np_ratings,neighbors):
    """
    :param userid : user id for which we wish to find candidate items    
    :return : I_u, candidates
    """
    
    # 1. Finding the set I_u of items already rated by user userid
    I_u = np_ratings[np_ratings[:, 0] == userid]
    I_u = I_u[:, 1]
    
    # 2. Taking the union of similar items for all items in I_u to form the set of candidate items
    c = set()
        
    for iid in I_u:    
        # add the neighbors of item iid in the set of candidate items
        c.update(neighbors[int(iid)])
        
    c = list(c)
    # 3. exclude from the set C all items in I_u.
    candidates = np.setdiff1d(c, I_u, assume_unique=True)
    
    return I_u, candidates


def similarity_with_Iu(c,I_u,neighbors,similarities):
    """
    compute similarity between an item c and a set of items I_u. For each item i in I_u, get similarity between 
    i and c, if c exists in the set of items similar to itemid.    
    :param c : itemid of a candidate item
    :param I_u : set of items already purchased by a given user    
    :return w : similarity between c and I_u
    """
    w = 0    
    for iid in I_u :        
        # get similarity between itemid and c, if c is one of the k nearest neighbors of itemid
        if c in neighbors[int(iid)] :
            w = w + similarities[int(iid), neighbors[int(iid)] == c][0]    
    return w


def rank_candidates(candidates,I_u,neighbors,similarities):
    """
    rank candidate items according to their similarities with i_u    
    :param candidates : list of candidate items
    :param I_u : list of items purchased by the user    
    :return ranked_candidates : dataframe of candidate items, ranked in descending order of similarities with I_u
    """
    
    # list of candidate items mapped to their corresponding similarities to I_u
    sims = [similarity_with_Iu(c,I_u,neighbors,similarities) for c in candidates]
    #candidates = label_encoder.inverse_transform(candidates)    
    mapping = list(zip(candidates, sims))
    
    ranked_candidates = sorted(mapping, key=lambda couple:couple[1], reverse=True)    
    return ranked_candidates

def topn_recommendation(userid,np_ratings,neighbors,similarities, N=10):
    """
    Produce top-N recommendation for a given user    
    :param userid : user for which we produce top-N recommendation
    :param n : length of the top-N recommendation list    
    :return topn
    """
    # find candidate items
    I_u, candidates = candidate_items(userid,np_ratings,neighbors)
    
    # rank candidate items according to their similarities with I_u
    ranked_candidates = rank_candidates(candidates,I_u,neighbors,similarities)
    
    # get the first N row of ranked_candidates to build the top N recommendation list
    topn = pd.DataFrame(ranked_candidates[:N], columns=['therapy','similarity_with_Iu'])    
    #topn = pd.merge(topn, data_p, on='CT', how='inner')    
    return topn


def predict(userid, itemid,data_p,np_ratings,neighbors,similarities):
    """
    Make rating prediction for user userid on item itemid    
    :param userid : id of the active user
    :param itemid : id of the item for which we are making prediction        
    :return r_hat : predicted rating
    """
    
    # Get items similar to item itemid with their corresponding similarities
    item_neighbors = neighbors[int(itemid)]
    item_similarities = similarities[int(itemid)]

    # get ratings of user with id userid
    uratings = np_ratings[np_ratings[:, 0] == userid]
    
    # similar items rated by item the user of i
    siru = uratings[np.isin(uratings[:, 1], item_neighbors)]
    scores = siru[:, 2]
    indexes = [np.where(item_neighbors == iid)[0][0] for iid in siru[:,1]]    
    sims = item_similarities[indexes]
    
    dot = np.dot(scores, sims)
    som = np.sum(np.abs(sims))
    mean, _ = normalize(data_p)
    mean_id=data_p.groupby(by='id', as_index=False)['successful'].mean()
    user_index=mean_id[mean_id["id"]==userid].index.values[0]

    if dot == 0 or som == 0:
        return mean[user_index]
    
    return dot / som


def topn_prediction(userid,data_p,np_ratings,neighbors,similarities):
    """
    :param userid : id of the active user    
    :return topn : initial topN recommendations returned by the function item2item_topN
    :return topn_predict : topN recommendations reordered according to rating predictions
    """
    # make top N recommendation for the active user
    topn = topn_recommendation(userid,np_ratings,neighbors,similarities)
    
    # get list of items of the top N list
    itemids = topn.therapy.to_list()
    
    predictions = []
    
    # make prediction for each item in the top N list
    for itemid in itemids:
        r = predict(userid,itemid,data_p,np_ratings,neighbors,similarities)
        
        predictions.append((itemid,r))
    
    predictions = pd.DataFrame(predictions, columns=['therapy','prediction'])
    
    # merge the predictions to topN_list and rearrange the list according to predictions
    topn_predict = pd.merge(topn, predictions, on='therapy', how='inner')
    topn_predict = topn_predict.sort_values(by=['similarity_with_Iu'], ascending=False)
    #topn_predict["therapy"]=label_encoder_therapy.inverse_transform(topn_predict["therapy"])    
    return topn, topn_predict

def test(test_list,therapies,label_encoder_therapy,data_p, np_ratings, neighbors, similarities):

  patient=[]
  condition=[]
  prediction_therapy=[]
  prediction_name=[]


  for i in test_list:
      topn, topn_predict=topn_prediction(userid=i,data_p=data_p,np_ratings=np_ratings,neighbors=neighbors,similarities=similarities)
      topn_predict["therapy"]=label_encoder_therapy.inverse_transform(topn_predict["therapy"])
      pred=pd.merge(topn_predict,therapies,on="therapy",how="left")[["therapy","name"]]
      patient.append(i.split("-")[0])
      condition.append(i.split("-")[1])
      prediction_therapy.append(pred["therapy"].to_list())
      prediction_name.append(pred["name"].to_list())
  pred=pd.DataFrame([patient,condition,prediction_therapy,prediction_name]).T
  pred.columns=["PatientId","condition","Recommended_Therapy","Recommended_Therapy_name"]
  results=pred.to_dict('records')

  with io.open('test_results.txt', 'w', encoding='utf-8') as f:
    f.write(json.dumps(results, ensure_ascii=False))
  print("test result is saved as test_results.txt")
  return results