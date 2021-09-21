# Import pandas library
import pandas as pd
import pickle as pkl

# Import warnings
import warnings
warnings.filterwarnings('ignore')

"""
This class prepares data by loading from source, dividing selected feature by 10, and then saving the new data.
"""

class PrepareData():

    # init method or constructor
    def __init__(self):
        self.__load_source_data()
        #self.feat_engg_cols = ['anaemia','sex','smoking','platelets','diabetes','high_blood_pressure','time']

    def __load_source_data(self):
        self.df = pd.read_csv("data/source/HF_dataset.csv")

    def reduce_feature_scale(self):
        """
        Function that divides selected features by 10
        """
        self.df = self.df.drop(['anaemia','sex','smoking','platelets','diabetes','high_blood_pressure','time'], axis = 1)

    def save_data_and_objects(self):
        self.df.to_csv("data/output/HFdata_final.csv", index=False)
        #pkl.dump(self.feat_engg_cols, open("model/feature_engg_cols.obj", 'wb'))
    

def main():
    prep_data = PrepareData()
    prep_data.reduce_feature_scale()
    prep_data.save_data_and_objects()

if __name__ == "__main__":
    main()
