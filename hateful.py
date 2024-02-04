import pandas as pd
import joblib
from class_data import *

model = joblib.load("model/combinedmodel.joblib")

def is_hateful(comment):
     df = pd.DataFrame([comment], columns=['text'])
     y_probabilities = model.predict_proba(df)[:, 1]

    # Set a custom threshold (e.g., 0.3)
     custom_threshold = 0.3

    # Apply the custom threshold to make predictions
     y_custom_predictions = (y_probabilities > custom_threshold).astype(int)
     if(y_custom_predictions == 1):
          return True
     else:
          return False

if __name__ == "__main__":
    comment = "i am disgusted by you. you are a disgrace"
    if(is_hateful(comment)):
        print("Hateful")
    else:
         print("Not Hateful")
    