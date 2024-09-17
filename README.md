# Laptop Price Prediction

This project involves predicting laptop prices in EURO using various machine learning models. It includes data preprocessing, model training, and predictions.

## Project Structure

- `train_data.csv`: The training dataset.
- `test_data.csv`: The test dataset for predictions.
- `param_dict.pickle`: A pickle file containing parameters for preprocessing.
- `finalized_model.sav`: A pickle file containing the trained LightGBM model.
- `prediction_results.csv`: Output file containing predictions.

- `train_test_split.ipynb`: A Python file designed for processing data for model training.
- `columns.py`: A Python file containing all nessecery dependencies of datasets.
- `model_best_hyperparameters.py`: Best parameters for prediction model.
- `train.py`: A Python file training LightGBM model.

- `predict.py`: A Python Output file containing !!!YOUR!!! predictions.

## Requirements

Make sure you have all the necessary libraries installed. 
You can install by opening folder of this project and running this command in Commnad Line or any IDLE:

    pip install -r requirements.txt


## Usage

## Training the Model

Ensure train_data.csv is available in the working directory.

Run the training script to preprocess the data, train the model, and save the parameters and model:

    python train_model.py

## Making Predictions

Rename your data to "test_data.csv" and replace current file in directory

Ensure test_data.csv is available in the working directory.

Run the prediction script to preprocess the test data and generate predictions:

    python predict.py

The predictions will be saved in prediction_results.csv.