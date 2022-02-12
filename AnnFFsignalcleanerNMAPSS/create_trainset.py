import pandas as pd
from pandas import DataFrame
import numpy as np
import h5py


def read_h5(filename):
    with h5py.File(filename, 'r') as hdf:
        # Development set
        W_dev = np.array(hdf.get('W_dev'))
        X_s_dev = np.array(hdf.get('X_s_dev'))
        X_v_dev = np.array(hdf.get('X_v_dev'))
        T_dev = np.array(hdf.get('T_dev'))
        Y_dev = np.array(hdf.get('Y_dev'))
        A_dev = np.array(hdf.get('A_dev'))

        # Varnams
        W_var = np.array(hdf.get('W_var'))
        X_s_var = np.array(hdf.get('X_s_var'))
        X_v_var = np.array(hdf.get('X_v_var'))
        T_var = np.array(hdf.get('T_var'))
        A_var = np.array(hdf.get('A_var'))

        # from np.array to list dtype U4/U5
        W_var = list(np.array(W_var, dtype='U20'))
        X_s_var = list(np.array(X_s_var, dtype='U20'))
        X_v_var = list(np.array(X_v_var, dtype='U20'))
        T_var = list(np.array(T_var, dtype='U20'))
        A_var = list(np.array(A_var, dtype='U20'))

        df_A = DataFrame(data=A_dev, columns=A_var)
        units = np.unique(df_A['unit'])
        # EOF (identify the number of cycle per unit)
        eof = {}
        for i in np.unique(df_A['unit']):
            eof[i] = len(np.unique(df_A.loc[df_A['unit'] == i, 'cycle']))

        # EOF (identify the number of cycle per healthy unit)
        eof_healthy = {}
        for i in np.unique(df_A['unit']):
            eof_healthy[i] = len(np.unique(df_A.loc[(df_A['unit'] == i) & (df_A['hs'] == 1), 'cycle']))

        df_W = DataFrame(data=W_dev, columns=W_var)

        df_X = pd.concat([df_A, df_W], axis=1)
        df_X['unit'] = pd.to_numeric(df_X['unit'], downcast='integer')
        df_X['alt'] = pd.to_numeric(df_X['alt'], downcast='integer')
        df_X['Fc'] = pd.to_numeric(df_X['unit'], downcast='integer')
        df_X['cycle'] = pd.to_numeric(df_X['cycle'], downcast='integer')
        df_X['hs'] = pd.to_numeric(df_X['hs'], downcast='integer')

        df_Y = DataFrame(data=X_s_dev, columns=X_s_var)
        df_Y = df_Y.astype('float32') # reduce the dimension of the dataset
        training_set = pd.concat([df_X, df_Y], axis=1)
        training_set = training_set.loc[training_set['hs'] == 1] # Take healthy elements

        # create the training set decimating and normalizing the cycles in a range from 0 to 1
        columns_name = training_set.columns.values.tolist()
        training_set_decimate = DataFrame(columns=columns_name)
        for i in units:
            cycle_per_unit = eof_healthy[i]
            for j in range(1, int(cycle_per_unit)):
                subset = training_set.loc[(training_set['unit'] == int(i)) & (training_set['cycle'] == j)]
                subset = subset[::10] # decimate
                subset_len = len(subset)
                dt = 1/subset_len
                cycle_normalized = np.arange(0, 1, dt)
                cycle_normalized = cycle_normalized[:subset_len]
                subset = subset.drop(['cycle'], axis=1)
                subset.insert(1, 'cycle', cycle_normalized)
                training_set_decimate = training_set_decimate.append(subset)

        train_X = training_set_decimate.iloc[:,0:8] # flight condition
        train_X = train_X.drop(['unit', 'hs'], axis=1)
        train_Y = training_set_decimate.iloc[:, 8:] # sensor

        # save dataset
        training_set_decimate.to_pickle("dataset/training_set_decimate.pkl")
        train_X.to_pickle("dataset/train_X.pkl")
        train_Y.to_pickle("dataset/train_Y.pkl")
        print(training_set_decimate)


if __name__ == '__main__':
    read_h5("C:/Users/agost/OneDrive/Documenti/Università/Magistrale/Manutenzione preventiva per la robotica/Progetto/D - PHM2021 Data  Challenge - Dataset/data_set/N-CMAPSS_DS04.h5")


