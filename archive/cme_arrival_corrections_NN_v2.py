#/opt/miniconda3/envs/tf_env/bin/python has all the required packages

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 1=hide INFO, 2=hide INFO+WARNING, 3=hide all

import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import polars
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

# helper functions
def read_time_from_file(filename):
    with open(filename, 'r') as f:
        last_line = f.readlines()[-1]
    last_line = last_line.split()
    time = last_line[0]
    #make it datetime object
    time = datetime.strptime(time, '%Y/%m/%dT%H:%M')
    return time

# NN model
def NN_model(input_shape):

    """
    makes and returns the object NN model
    """
    # input layer
    ip_layer = Input(shape = (input_shape,))

    # hidden layer 1
    dl1 = Dense(256, activation='relu')(ip_layer)
    b1 = BatchNormalization()(dl1)
    d1 = Dropout(0.3)(b1)

    # hidden layer 2
    dl2 = Dense(128, activation="relu")(d1)
    b2 = BatchNormalization()(dl2)
    d2 = Dropout(0.2)(b2)

    # hidden layer 3
    dl3 = Dense(64, activation="relu")(d2)
    b3 = BatchNormalization()(dl3)
    d3 = Dropout(0.2)(b3)

    # hidden layer 4
    dl4 = Dense(32, activation="relu")(d3)
    b4 = BatchNormalization()(dl4)
    d4 = Dropout(0.2)(b4)

    # hidden layer 5
    dl5 = Dense(16, activation="relu")(d4)
    b5 = BatchNormalization()(dl5)
    d5 = Dropout(0.2)(b5)

    # the output
    output = Dense(1, activation="relu")(d5)

    # make and return model
    model = Model(inputs=ip_layer, outputs=output)
    return model

N = 100  # target samples per time
def interp_group(g: polars.DataFrame) -> polars.DataFrame:
    # x-grid to interpolate to (use group min/max in case they’re not 1..21)
    x_new = np.linspace(float(g["Ensemble_member"].min()), float(g["Ensemble_member"].max()), N)

    # make sure x is strictly increasing (drop duplicates if any)
    g = g.sort("Ensemble_member").unique(subset=["Ensemble_member"], keep="first")

    x = g["Ensemble_member"].to_numpy()
    out = {
        "Time_since_eruption": np.full(N, float(g["Time_since_eruption"][0])),
        "Ensemble_member": x_new.round(1),
        "EA_diff_A": np.interp(x_new, x, g["EA_diff_A"].to_numpy()),
        "EA_diff_B": np.interp(x_new, x, g["EA_diff_B"].to_numpy()),
        "Travel_time_y": np.interp(x_new, x, g["Travel_time_y"].to_numpy()),
    }
    return polars.DataFrame(out)
    
# to make sure we are using both A and B
CME_num_which_craft = {'01_2010-04-03': 'Both', '02_2010-05-23': 'Both', '03_2010-08-01' : 'Both', '04_2011-09-06': 'Both', '05_2011-09-13': 'Both', '06_2011-10-22': 'Both', '07_2012-01-19': 'Both', '09_2012-06-14': 'Both', '10_2012-07-03': 'Both','11_2012-07-12': 'Both', '12_2012-09-27': 'Both', '13_2012-10-05': 'Both', '15_2013-06-30': 'Both'}
outfile = '00ML_NN_results_AB.txt'

# Create a df with columns CME_num, Actual_TT, Seed_TT, ML_TT, Seed_error, ML_error
df_out = pd.DataFrame(columns=['CME_num', 'Actual_TT', 'Seed_TT', 'ML_TT', 'Seed_error', 'ML_error'])

#open text file and write column names
with open(outfile, 'w') as f:
    f.write('CME_num,Actual_TT,Seed_TT,ML_TT,Seed_error,ML_error\n')

for CME_num in ['01_2010-04-03', '02_2010-05-23', '03_2010-08-01', '04_2011-09-06', '05_2011-09-13', '06_2011-10-22', '07_2012-01-19', '09_2012-06-14', '10_2012-07-03','11_2012-07-12', '12_2012-09-27', '13_2012-10-05', '15_2013-06-30']:

    which_craft = CME_num_which_craft[CME_num]

    # read in the data
    data = polars.read_csv('../../CMEs/ML/Train_'+CME_num+'_AB.txt')
    data = data[["Ensemble_member", "Time_since_eruption", "EA_diff_A", "EA_diff_B", "Travel_time_y"]]

    # start from your existing DataFrame: data
    data = data.sort("Time_since_eruption")

    # interpolate the data: A total of 100 points at each time step
    # data = (data.group_by("Time_since_eruption", maintain_order=True).map_groups(interp_group))

    # Features
    if which_craft == "A":
        X = data[["Time_since_eruption", "EA_diff_A"]]
    elif which_craft == "B":
        X = data[["Time_since_eruption", "EA_diff_B"]]
    elif which_craft == "Both":
        X = data[["Time_since_eruption", "EA_diff_A", "EA_diff_B"]]
    y = data[["Travel_time_y"]]

    # make them numpy arrays
    X = X.to_numpy()
    y = y.to_numpy()

    # Standard scalar for the features
    scalar_X = StandardScaler()
    X = scalar_X.fit_transform(X) # (1) compute the mean and std, and (2) then scale X_train

    # MinMax scalar for the targets
    scalar_y = MinMaxScaler(feature_range=(0.1, 0.9)) 
    y = scalar_y.fit_transform(y)

    cme_data = (
        data
        .filter(polars.col("Ensemble_member") == 1.0)
        .with_columns(
            polars.lit(0.0).alias("EA_diff_A"),
            polars.lit(0.0).alias("EA_diff_B"),
        )
    )

    cme_data = cme_data[-1]

    # make model
    model = NN_model(int(X.shape[1]))

    # compile with loss = mse, optimizer = adam, and metrics = "mae"
    model.compile(loss = "mse", 
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    # get the callbacks
    callbacks = [EarlyStopping(monitor = "val_loss", patience=15, restore_best_weights=True), 
                ReduceLROnPlateau(monitor = "val_loss", patience=15, factor = 0.1, min_lr = 1e-6)]
    
    # summary of the NN model
    model.summary()

    # Train the model
    model_trained = model.fit(
        X,
        y,
        epochs = 100,
        verbose = 1,
        callbacks = callbacks,
        shuffle = True,
        validation_split = 0.1
    )

    address = '../../CMEs/'

    #Get the eruption time of the CME
    Erupt_file = address+CME_num+"/Erupt_time.txt"
    CME_Erupt_time = read_time_from_file(Erupt_file)

    #Get the arrival time of the CME
    Arrival_file = address+CME_num+"/Arrival_time.txt"
    CME_Arrival_time = read_time_from_file(Arrival_file)

    #Calculate the travel time in hours
    Obs_CME_travel_time = CME_Arrival_time - CME_Erupt_time
    Obs_CME_travel_time = Obs_CME_travel_time.total_seconds() / 3600

    #seperating training and target variables
    X_actual = cme_data[["Time_since_eruption", "EA_diff_A","EA_diff_B"]]
    #X_actual = X_actual[X_actual['Time_since_eruption'] == X_actual['Time_since_eruption'].max()]
    if which_craft == 'A':
        X_actual = cme_data[["Time_since_eruption", "EA_diff_A"]]
    elif which_craft == 'B':
        X_actual = cme_data[["Time_since_eruption", "EA_diff_B"]]
    y_actual = cme_data[["Travel_time_y"]]
    import polars as pl

    y_actual = y_actual.with_columns(
        polars.lit(Obs_CME_travel_time).alias("Travel_time_y")
    )

    X_actual = X_actual.to_numpy()
    y_actual = y_actual.to_numpy()

    # make the prediction based on the model:
    prediction = model.predict(X_actual)
    
    # Transform back to original scale using the scalar_y object
    actual_predictions = scalar_y.inverse_transform(prediction)

    # Subset at Ensemble_member == 11.0 (Polars way; with nearest fallback)
    data_subset = data.filter(polars.col("Ensemble_member") == 11.0)
    if data_subset.height == 0:
        data_subset = (
            data
            .with_columns((polars.col("Ensemble_member") - 11.0).abs().alias("dist"))
            .sort("dist")
            .head(1)
            .drop("dist")
        )

    # Get scalar from Polars
    travel_time = float(data_subset.get_column("Travel_time_y")[0])

    seed_error = abs(travel_time - y_actual.mean())
    ML_error = abs(y_actual.mean() - actual_predictions.mean())
    print('CME_num', CME_num, 'Actual_TT', Obs_CME_travel_time, 'Seed_TT', travel_time, 'ML_TT', actual_predictions, 'Seed_error', seed_error, 'ML_error', ML_error)
    #Write in already opened file
    with open(outfile, 'a') as f:
        f.write(CME_num+','+str(round(Obs_CME_travel_time,2))+','+str(round(travel_time,2))+','+str(round(actual_predictions.mean(),2))+','+str(round(seed_error,2))+','+str(round(ML_error,2))+'\n')
    #append to df_out
    new_row = pd.DataFrame([{'CME_num': CME_num, 'Actual_TT': Obs_CME_travel_time, 'Seed_TT': travel_time, 'ML_TT': actual_predictions.mean(), 'Seed_error': seed_error, 'ML_error': ML_error}])
    df_out = pd.concat([df_out, new_row], ignore_index=True)

print(df_out)

#Average of Seed_error and ML_error
print("Average Seed Error: ", df_out["Seed_error"].mean())
print("Average ML Error: ", df_out["ML_error"].mean())

#Print these to outfile
with open(outfile, 'a') as f:
    f.write("Average Seed Error: "+str(round(df_out["Seed_error"].mean(),2))+'\n')
    f.write("Average ML Error: "+str(round(df_out["ML_error"].mean(),2))+'\n')