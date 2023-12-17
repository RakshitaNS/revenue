import sys
import pandas as pd
#from src_.exception import CustomException
# from src.components.data_transformation import DataTransformation
# from src.components.data_transformation import DataTransformationConfig
# from src.components.src_.utils import load_object
import os
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        model_path=os.path.join('artifacts','model.pkl')
        preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
        print("Before Loading")
        model=load_object(file_path=model_path)
        preprocessor=load_object(file_path=preprocessor_path)
        print("After Loading")
        
        # le = LabelEncoder()
        # preprocessor=le.fit_transform(preprocessor)
        pre=StandardScaler()
        data_scaled=pre.fit_transform(preprocessor)
        data_scaled=data_scaled.reshape(-1, 1)
        x=self.predict(data_scaled)
        preds=model(x)
        return preds
     
def load_object(file_path):
        if os.path.exists(file_path):
             with open(file_path, "rb") as file_obj:
                  return pickle.load(file_obj)
    

class CustomData:
    def __init__(  self,
        Id: int,
        Name: str,
        Franchise: str,
        Category: str,
        City: str,
        No_Of_Item: int,
        Order_Placed: float):

        self.Id = Id

        self.Name = Name

        self.Franchise = Franchise

        self.Category = Category

        self.City = City

        self.No_Of_Item = No_Of_Item

        self.Order_Placed = Order_Placed

    def get_data_as_data_frame(self):

            custom_data_input_dict = {
                "Id": [self.Id],
                "Name": [self.Name],
                "Franchise": [self.Franchise],
                "Category": [self.Category],
                "City": [self.City],
                "No_Of_Item": [self.No_Of_Item],
                "Order_Placed": [self.Order_Placed],
            }

            return pd.DataFrame(custom_data_input_dict)