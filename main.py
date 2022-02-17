from model_functions import *

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


test_data=pd.read_csv('Dataset/datasetB_cases.txt', delimiter = "\t")
test_data=test_data.reset_index()
test_data["PatientID"]=test_data["index"]
test_data.drop("index",inplace=True,axis=1)
test_data.columns=["id","condition"]
test_data=test_data.merge(df[["id","condition","kind"]],on=["id","condition"],how="left").drop_duplicates()
test_data["id"]=test_data["id"].astype(str)+"-"+test_data["kind"]
test_data.fillna("9999999-Cond99999",inplace=True)
test_list=test_data["id"].to_list()

# recommend therapies for test data
test(test_list,therapies,label_encoder_therapy,data_p, np_ratings, neighbors, similarities)  
