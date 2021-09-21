# import Flask class from the flask module
from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas

# Create Flask object to run
app = Flask(__name__)

@app.route('/')
@app.route('/index')
def home():
    return "Hi, Welcome to Flask!!"

########## LOADING OBJECTS. THIS WILL BE DONE ONLY ONCE ##########
# print ("Loading model and objects...")
# # Loading model
HF_pred_file = open('model/gbc_finalmodel.mdl', 'rb')
HF_pred_model = pickle.load(HF_pred_file)
HF_pred_file.close()

##################################################################

# Render Concrete mixture input page
@app.route('/input')
def input():
    return render_template('input.html')

# This function will be called when the input page is submitted
@app.route('/predict', methods=["POST"])
def predict():

    # Enter into this snippet of the code only if the method is POST.
    if request.method == "POST":

        # Get values from browser
        input_dict = request.form.to_dict()

        # Extract keys and values from the input dictionary object
        form_keys =list(input_dict.keys())
        form_values =list(input_dict.values())

        # Convert them to float as they will be in String format
        form_values = map(float, form_values)

        # Construct the dictionary object with the existing keys and float values
        input_dict = dict(zip(form_keys, form_values))


        # Construct the dataframe out of the dictionary object
        input_df = pandas.DataFrame(input_dict, index=[0])
        print ("Input values: \n", input_df)


        # Pass the dataframe object to loaded ML model and do prediction
        DEATH_EVENT_predicted = str(round(HF_pred_model.predict(input_df)[0], 2))
        print ("Predicted Heart Failure: ", DEATH_EVENT_predicted)

        return render_template('results.html', DEATH_EVENT_predicted=DEATH_EVENT_predicted)


if __name__ == "__main__":
    print("**Starting Server...")


    # Run Server
    app.run()
