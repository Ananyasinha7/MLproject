import os # for path operations
import sys # for exception handling
from dataclasses import dataclass 

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train,y_train,X_test,y_test=(train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1],)

            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbors":KNeighborsRegressor(),
                "XGB Regressor":XGBRegressor(),
                "CatBoost Regressor":CatBoostRegressor(verbose=False),
                "AdaBoost Regressor":AdaBoostRegressor(),
                "Gradient Boosting":GradientBoostingRegressor()
            }

            params={
                "Decision Tree":{
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter': ['best', 'random'],
                    # 'max_features': [ 'sqrt', 'log2'],
                },
                "Random Forest":{
                    'n_estimators':[8,16,32,64,128,256],
                    #criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    #'max_features': ['sqrt','log2',None],
                },
                "Linear Regression":{},
                "K-Neighbors":{
                    'n_neighbors':[5,7,9,11],
                    #'weights':['uniform','distance'],
                    #'algorithm':['auto','ball_tree','kd_tree','brute']
                },
                "XGB Regressor":{
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "CatBoost Regressor":{
                    'depth':[6,8,10],
                    'learning_rate':[0.01,0.05,0.001],
                    'iterations':[30,50,100],
                },
                "AdaBoost Regressor":{
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    #'loss':['linear','square','exponential'],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    'n_estimators':[8,16,32,64,128,256],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    #'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'
                    #'max_features':['auto','sqrt','log2'
                }
            }






            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test, models=models,params=params)

            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test, predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)