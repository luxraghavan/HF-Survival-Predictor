# Import pandas library
import pandas as pd

# Import pickle library
import pickle as pkl

# Import GradientBoostingClassifier library
from sklearn.ensemble import GradientBoostingClassifier

# Import warnings
import warnings
warnings.filterwarnings('ignore')

class TrainFinalModel():

    def train_gbc_finalmodel(self):

        # Load dataset
        df = pd.read_csv("data/output/HFdata_final.csv")

        # Generate a Gradient Boosting Classifier object
        gbc_model = GradientBoostingClassifier(max_depth=2, random_state=1)

        # Prepare train data object specific to GBC
        gbc_model.fit(df.drop('DEATH_EVENT', axis = 1), df['DEATH_EVENT'])
        #gbc_model.fit(df.drop('DEATH_EVENT', axis = 1), df['DEATH_EVENT'])

        # To persist the base model into hard-disk, uncomment the below line
        pkl.dump(gbc_model, open("model/gbc_finalmodel.mdl", 'wb'))

        print("GradientBoostingClassifier final model generated and persisted!!")



# Class ends here
def main():
    train_final_model = TrainFinalModel().train_gbc_finalmodel()

if __name__ == "__main__":
    main()
