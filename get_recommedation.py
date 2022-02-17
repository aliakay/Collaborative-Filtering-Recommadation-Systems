import argparse
from model_functions import *

argparser = argparse.ArgumentParser(description='hyper-parameters')

argparser.add_argument('--patient',type=int, default=6, help='Patient id')
argparser.add_argument('--condition',type=str, default="Cond248", help='Condition id')
arg = argparser.parse_args()

#load dataset
f = open('Dataset/datasetB.json', 'r')
dataset = json.loads(f.read())
therapies=pd.DataFrame.from_dict(pd.json_normalize(dataset["Therapies"]), orient='columns')
therapies.columns=["therapy","name","type"]
conditions=pd.DataFrame.from_dict(pd.json_normalize(dataset["Conditions"]), orient='columns')
conditions.columns=["conditions","name","type"]

#load pre-processed dataset
df=pd.read_csv("Dataset/datamining_data.csv")
df["id"]=df.id.astype(int)

#create dataset in order to create the matrix
data_p=df[["id","kind","therapy","successful"]]

#create id == "patientid-condition"
data_p["id"] = data_p["id"].astype(str) + "-" + data_p["kind"]
data_p=data_p[["id","therapy","successful"]]

#get normalize rating of succesful column
mean, norm_ratings = normalize(data_p)

# label_encoder object knows how to understand word labels.
label_encoder_therapy = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
norm_ratings['therapy']= label_encoder_therapy.fit_transform(norm_ratings['therapy'])

#norm_ratings=norm_ratings.sort_values("therapy",ascending=True)
norm_ratings=norm_ratings[["id","therapy","successful","norm_successful"]]
np_ratings = norm_ratings.to_numpy()


#create the succesful rating matrix using normalized rates and get similarities
R = rating_matrix(norm_ratings,"norm_successful")
model = create_model(R, metric="cosine")
similarities, neighbors = nearest_neighbors(R, model)

patient_id=arg.patient
cond_id=arg.condition

test_id=str(patient_id)+ "-" +cond_id

topn, topn_predict = topn_prediction(userid=test_id,data_p=data_p,np_ratings=np_ratings,neighbors=neighbors,similarities=similarities)
topn_predict["therapy"]=label_encoder_therapy.inverse_transform(topn_predict["therapy"])
patient=[]
condition=[]
prediction_therapy=[]
prediction_name=[]

pred=pd.merge(topn_predict,therapies,on="therapy",how="left")[["therapy","name"]]
patient.append(arg.patient)
condition.append(arg.condition)
prediction_therapy.append(pred["therapy"].to_list())
prediction_name.append(pred["name"].to_list())
pred=pd.DataFrame([patient,condition,prediction_therapy,prediction_name]).T
pred.columns=["PatientId","condition","Recommended_Therapy","Recommended_Therapy_name"]
results=pred.to_dict('records')

with io.open("patient{}_{}_recommadations.txt".format(patient_id, cond_id), 'w', encoding='utf-8') as f:
    f.write(json.dumps(results, ensure_ascii=False))

print("Recommadation saved.")



