import pandas as pd
from scipy.ndimage import label
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import  r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pickle
import sys
class REGRESSION:
    def __init__(self):
        self.slr_data = None
        self.mlr_data = None
        self.slr_reg = None
        self.mlr_reg = None
    def load_data(self):
        try:
            self.slr_data = pd.read_csv("Salary_Data.csv")
            self.mlr_data = pd.read_csv("50_Startups.csv")
            print(f"Simple Linear Regreesion:\n{self.slr_data.head()}")
            print(f"Multiple Linear Regreesion:\n{self.mlr_data.head()}")
        except Exception:
            exc_type, exc_line, exc_msg = sys.exc_info()
            print(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")
    def slr(self):
        try:
            slr_X = self.slr_data.iloc[:,0]
            slr_y = self.slr_data.iloc[:,1]

            # splitting the data for training and testing
            slr_X_train, slr_X_test, slr_y_train, slr_y_test = train_test_split(slr_X,slr_y, test_size = 0.3, random_state= 42)
            slr_trained_data = pd.DataFrame()
            slr_trained_data["Experience"] = slr_X_train
            slr_trained_data["Salary"] = slr_y_train
            slr_test_data = pd.DataFrame()
            slr_test_data["Experience"] = slr_X_test
            slr_test_data["Salary"] = slr_y_test

            # Model
            self.slr_reg = LinearRegression()
            slr_X_train = slr_X_train.values.reshape(-1,1)
            slr_X_test = slr_X_test.values.reshape(-1, 1)
            self.slr_reg.fit(slr_X_train, slr_y_train)

            #train predictions
            slr_y_train_pred = self.slr_reg.predict(slr_X_train)
            slr_trained_data["Predicted_Values_Model"] = slr_y_train_pred
            print(f"The training data with predictions from the model are:\n{slr_trained_data}")
            print(f"The accuracy of the slr Model is:{r2_score(slr_y_train, slr_y_train_pred)}")
            print(f"The Average loss of the model is:{mean_squared_error(slr_y_train, slr_y_train_pred)}")

            #Test predictions
            slr_y_test_pred = self.slr_reg.predict(slr_X_test)
            slr_test_data["Predicted_Values_Model"] = slr_y_test_pred
            print(f"The training data with predictions from the model are:\n{slr_test_data}")
            print(f"The accuracy of the slr Model is:{r2_score(slr_y_test, slr_y_test_pred)}")
            print(f"The Average loss of the model is:{mean_squared_error(slr_y_test, slr_y_test_pred)}")
            #training data Visuallization
            plt.figure(figsize=(5,3))
            plt.xlabel("X_train Values")
            plt.ylabel("y_train and prediction Values")
            plt.scatter(slr_X_train, slr_y_train, color="r", marker="s", label="Actual values")
            plt.plot(slr_X_train, slr_y_train_pred, color="b", marker="*", label="Predicted values")
            plt.legend(loc=0)
            plt.show()
            #test data Visuallization
            plt.figure(figsize=(5,3))
            plt.xlabel("X_test Values")
            plt.ylabel("y_test and prediction Values")
            plt.scatter(slr_X_test, slr_y_test, color="r", marker="s", label="Actual values")
            plt.plot(slr_X_test, slr_y_test_pred, color="b", marker="*", label="Predicted values")
            plt.legend(loc=0)
            plt.show()
        except Exception:
            exc_type, exc_msg, exc_line = sys.exc_info()
            print(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")
    def mlr(self):
        try:
            #gather the data
            print(self.mlr_data["State"].unique())
            self.mlr_data["State"] = self.mlr_data["State"].map({"New York":1, "California":2, "Florida":3})
            mlr_X = self.mlr_data.drop("Profit", axis = 1)
            mlr_y = self.mlr_data["Profit"]
            print(mlr_X)
            #splitting the data
            mlr_X_train, mlr_X_test, mlr_y_train, mlr_y_test = train_test_split(mlr_X, mlr_y, test_size=0.3, random_state=42)
            # print(mlr_X_train)
            #Model
            print(mlr_X_train)
            self.mlr_reg = LinearRegression()
            print(mlr_X_train)
            self.mlr_reg.fit(mlr_X_train, mlr_y_train)
            print("Training is done and model is ready to predict.")
            #training prediction
            mlr_y_train_pred = self.mlr_reg.predict(mlr_X_train)
            mlr_trained_data = pd.DataFrame()
            mlr_trained_data = mlr_X_train
            mlr_trained_data["Profit"] = mlr_y_train
            mlr_trained_data["Predicted_values"] = mlr_y_train_pred
            print(f"The trained data with the prediction values:\n{mlr_trained_data}")
            print(f"The Accuracy of the model on trained.\n{r2_score(mlr_y_train, mlr_y_train_pred)*100}")
            print(f"The Average Loss of the Model.{mean_squared_error(mlr_y_train, mlr_y_train_pred)}")
            #testing prediction
            mlr_y_test_pred = self.mlr_reg.predict(mlr_X_test)
            mlr_test_data = pd.DataFrame()
            mlr_test_data = mlr_X_test
            mlr_test_data["Profit"] = mlr_y_test
            print(f"The Accuracy of the model on trained.\n{r2_score(mlr_y_test, mlr_y_test_pred)*100}")
            print(f"The Average Loss of the Model.{mean_squared_error(mlr_y_test, mlr_y_test_pred)}")
            mlr_test_data["Predicted_Values"] = mlr_y_test_pred
            print(f"The test data with the prediction values:\n{mlr_test_data}")
        except Exception:
            exc_type, exc_msg, exc_line = sys.exc_info()
            print(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")


    def pickle_file(self):
            try:
                with open("slr_model.pkl", 'wb') as f:
                    pickle.dump(self.slr_reg,f)
                with open("mlr_model.pkl", 'wb') as f1:
                    pickle.dump(self.mlr_reg, f1)
            except Exception:
                exc_type, exc_msg, exc_line = sys.exc_info()
                print(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")
reg = REGRESSION()
reg.load_data()
reg.slr()
reg.mlr()
reg.pickle_file()