import sys
import pandas as pd
import os
from src.elu.exception import CustomException
from src.elu.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("src\\components\\artifacts","model.pkl")
            preprocessor_path=os.path.join('src\\components\\artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
    

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