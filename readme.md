# Therapy Recommadation Systems for Patients 

- University of Trento
- Data Mining Prof.Velegrakis
- Ali Akay [224414]
- MSc. Artificial Intelligence Systems

## Project Detail

Project contains three part;
- Data extraction-preparation
- Data Preprocessing
- Recommadation Systems Model

Please check the code in notebooks and run the code using instruction below.

You can find the result of the test dataset named "test_results.txt"

There are two dataset folder.
- Extracted Dataset: It is used in order to create dataset in json format after scrapping 
- Dataset: It is given by professor.
- Dataset/datamining_data.csv: It is generated in order to use for the model which is converted from json to dataframe.


### Requirements

Please install the libraries.If you already have this libraries you can skip this part.

- sklearn
- scipy
- json
- numpy
- pandas
- argparse

### Generate Recommadation

+ You can generate the recommadation for the test dataset given by professor and it will save a test_results.txt.

```python
$ python3 main.py
```
+ You can also generate recommadation for indivudial patients for conditions by changing the parameters.

```python
$ python3 get_recommedation.py --patient 0 --condition "Cond240"
```


