#init cell
#obserwujemy przebiegi sygnałów aby namierzyć outliery
import matplotlib.pyplot as plotter
def plot_column(data, col_num):
    %matplotlib qt
    figManager = plotter.get_current_fig_manager()
    figManager.window.showMaximized()
    plotter.plot(data.values[:, col_num], linewidth=0.5)
	
#obserwujemy przebiegi sygnałów aby namierzyć outliery
import matplotlib.pyplot as plotter
def plot_column2(data, col_num):
    %matplotlib qt
    figManager = plotter.get_current_fig_manager()
    figManager.window.showMaximized()
    plotter.plot(data[:, col_num], linewidth=0.5)
    

import numpy as np
def sim(model, test_set, set_num):
    col = len(test_set[set_num][0])
    y_serie = []
    for i in range(0, len(test_set[set_num])):
        if i == 0:
            sample = np.copy(test_set[set_num][0:1, 0:col-1])
            y = model.predict(sample)
        else:
            sample = np.copy(test_set[set_num][i:i+1, 0:col-1])
            sample[0, -1] = y
            y = model.predict(sample)
        y_serie.append(y)
    y_serie=np.array(y_serie)
    return y_serie


def sim_with_refreshes(model, test_set
                       ,set_num, refr_period):
    col = len(test_set[set_num][0])
    y_serie = []
    for i in range(0, len(test_set[set_num])):
        if i == 0 or i%refr_period == 0:
            sample = np.copy(test_set[set_num][i:i+1, 0:col-1])
            y = model.predict(sample)
        else:
            sample = np.copy(test_set[set_num][i:i+1, 0:col-1])
            sample[0, -1] = y
            y = model.predict(sample)
        y_serie.append(y)
    y_serie=np.array(y_serie)
    return y_serie


def eval_prediction(y_pred, y_target, do_plot):
    from sklearn.metrics import mean_squared_error as mse
    if do_plot == 1:

        plt.subplot(2, 1, 1)
        plt.plot(y_pred[:, 0], label='Prediction', lw=1)
        plt.plot(y_target, label='Target', lw=1)
        plt.legend()
        plt.grid(b=True, which='both', axis='y')

        residuals = np.copy(y_pred[:, 0])
        for i in range(0, len(residuals)):
            residuals[i] = residuals[i] - y_target[i]
        residuals = residuals.flatten()

        plt.subplot(2, 1, 2)
        plt.fill_between(range(0, len(residuals))
                             ,residuals, label='Residuals', lw=1)
        plt.legend()
        plt.grid(b=True, which='both', axis='y')
        plt.show()
    
    score = mse(y_pred[:, 0], y_target)
    print(score)
    return score
    
def multi_train_NLP(structs_list, patience, batch_size, 
                    epochs, train_data, test_data, set_num
                   ,criterion, single_times, dropout_2list):
    import sys
    bck_stdout = sys.stdout
    sys.stdout = open('log.txt', 'w')

    from sklearn.metrics import mean_squared_error as mse
    best_score=1
    
    struct_num = 0
    for structure in structs_list:
        print(f'Trenuje siec o struturze {structure}')
        curr_model = train_MLP_Ntimes(N=single_times
                                     ,structure=structure
                                     ,patience=patience
                                     ,batch_size=batch_size
                                     ,epochs=epochs
                                     ,set_num=set_num
                                     ,train_data=train_data
                                     ,test_data=test_data
                                     ,debug_log='no'
                                     ,criterion=criterion
                                     ,dropout_list=dropout_2list[struct_num])
    
        y = sim(curr_model, test_data, set_num)
        
        if criterion == 'mse':
            curr_score = mse(y[:, 0], test_data[set_num][:, -1])
        if criterion == 'max':
            residuals = np.copy(y[:, 0])
            for i in range(0, len(residuals)):
                residuals[i] = residuals[i] - test_data[set_num][i, -1]
            curr_score = np.max(residuals)      
            
        print(f'Wynik biezącej sieci: {curr_score}')
    
        if curr_score < best_score:
            best_model = curr_model
            best_score = curr_score
            
        struct_num = struct_num + 1
    sys.stdout = bck_stdout
    
    import winsound
    filename = 'jasny chuj.wav'
    winsound.PlaySound(filename, winsound.SND_FILENAME)
    
    return(best_model)

def train_MLP_Ntimes(N, structure, patience, batch_size, epochs
                    ,train_data, test_data, set_num
                    ,debug_log, criterion, dropout_list):
    best_score = 1
    nets_trained = 0
    
    if debug_log == 'yes':
        import sys
        bck_stdout = sys.stdout
        sys.stdout = open('log.txt', 'w')
    
    while nets_trained < N:
        model = train_NLP(structure=structure
                         ,patience=patience
                         ,batch_size=batch_size
                         ,epochs=epochs
                         ,train_data=train_data
                         ,dropout_list=dropout_list)
        y = sim(model, test_data, set_num)
        
        if criterion == 'mse':
            score = eval_prediction(y, test_set[set_num][:, -1], 0)
        if criterion == 'max':
            residuals = np.copy(y[:, 0])
            for j in range(0, len(residuals)):
                residuals[j] = residuals[j] - \
                test_data[set_num][j, -1]
            score = np.max(np.abs(residuals))
        print(f"Wynik sieci nr {nets_trained+1}: {score}")
        
        if score < best_score:
            best_score = score
            best_model = model
        
        del model
        nets_trained = nets_trained + 1
    
    import winsound
    filename = 'sound.wav'
    winsound.PlaySound(filename, winsound.SND_FILENAME)
    
    if debug_log == 'yes':
        sys.stdout = bck_stdout
    return best_model

def train_NLP(structure, patience,batch_size
              ,epochs, train_data, dropout_list):
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout
    from keras.callbacks import EarlyStopping
    from keras import regularizers
    from keras.optimizers import SGD
    
    inputs_count = len(train_data[0]) - 1
    layer_num = 1
    model = Sequential()
    
    for layer_size in structure:
        if layer_num == 1:
            model.add(Dense(layer_size, input_dim=inputs_count, 
                            activation='tanh'))
            model.add(Dropout(dropout_list[layer_num - 1]))
            
        else:
            model.add(Dense(layer_size, activation='tanh'))
            model.add(Dropout(dropout_list[layer_num - 1]))
            
        layer_num = layer_num + 1
    model.add(Dense(1, activation = 'linear'))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    es = EarlyStopping(monitor='val_loss', mode='min', 
                   verbose=1, patience=patience)

    model.fit(train_data[:, :-1],
              train_data[:, -1], batch_size=batch_size,
              epochs=epochs, validation_split = 0.35,
              callbacks=[es], workers=8, use_multiprocessing=1)
    return model


def train_SVR(C, epsilon, train_data):
    from sklearn.svm import SVR as svr
    model = svr(kernel='rbf', gamma='scale', C=C, epsilon=epsilon)
    model.fit(train_data[:, :-1], train_data[:, -1])
    
    return model

def multi_train_SVR(C_eps_pairs
                   ,train_data
                   ,test_data
                   ,set_num
                   ,criterion):
    import sys
    bck_stdout = sys.stdout
   # sys.stdout = open('log.txt', 'w')

    from sklearn.metrics import mean_squared_error as mse 
    best_score = 1

    for pair in C_eps_pairs:
        print(f'Trenuje SVR o C={pair[0]} i epsilon={pair[1]}')
        curr_model = train_SVR(C=pair[0]
                               ,epsilon=pair[1]
                               ,train_data=train_data)
    
        y = sim(curr_model, test_data, set_num)
        y = y.reshape(-1, 1)
        
        if criterion == 'mse':
            curr_score = mse(y[:, 0], test_data[set_num][:, -1])
        if criterion == 'max':
            residuals = np.copy(y[:, 0])
            for i in range(0, len(residuals)):
                residuals[i] = residuals[i] - test_data[set_num][i, -1]
            curr_score = np.max(residuals)
        
    
        print(f'Wynik biezacego SVRa: {curr_score}')
    
        if curr_score < best_score:
            best_score = curr_score
            best_model = curr_model

    sys.stdout = bck_stdout  
    return best_model

def train_forest(trees, alpha
                ,train_data):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=trees
        ,ccp_alpha=alpha
        ,max_features=1
        ,verbose=False
    )
    
    model.fit(train_data[:,:-1], train_data[:,-1])
    return model


def train_forest_nTimes(trees, alpha, n
                       ,train_data, test_data, test_num):
    best_score = 1
    for i in range(0, n):
        model = train_forest(trees, alpha
                            ,train_data)
        
        if n == 1:
            return model
        
        y = sim(model, test_data, test_num)
        score = eval_prediction(y, test_data[test_num][:, -1], 0)
        
        if score < best_score:
            best_score = score
            best_model = model
        
    return best_model


def train_multi_forest(trees, alpha
                      ,n, train_data, test_data, test_num):
    params=[]
    params.append(trees)
    params.append(alpha)
    
    best_score = 1
    import itertools
    for par_el in itertools.product(*params):
        print(f'Trenuje las o parametrach:')
        print(f'Liczba drzew: {par_el[0]}')
        print(f'CP dla pruningu: {par_el[1]}')
        
        model = train_forest_nTimes(par_el[0], par_el[1]
                                   ,n,train_data, test_data, test_num)
        
        y = sim(model, test_data, test_num)
        score = eval_prediction(y, test_data[test_num][:,-1], 0)
        
        if score < best_score:
            best_score = score
            best_model = model
            
    return best_model


def forest_obj_fun(x, train_data, test_data):
    import math
    model = train_forest_nTimes(
        trees=math.floor(x[0])
        ,depth=math.floor(x[1])
        ,split=math.floor(x[2])
        ,n=5
        ,train_data=train_data
        ,test_data=test_data
        ,test_num=9
    )
    
    col = 9
    y = sim(model, test_set_trim, col)
    score = eval_prediction(y, test_set_trim[col][:, -1], 0)
    
    return score



def train_adaBoost(trees, lr, depth
                ,train_data):
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.tree import DecisionTreeRegressor
    estimator = DecisionTreeRegressor(
        max_depth=depth
        ,max_features=1
    )
    
    model = AdaBoostRegressor(
        base_estimator=estimator
        ,n_estimators=trees
        ,learning_rate=lr
        ,loss='square'
    )
    
    model.fit(train_data[:,:-1], train_data[:,-1])
    return model


def train_adaBoost_nTimes(trees, lr, depth, n
                       ,train_data, test_data, test_num):
    best_score = 1
    for i in range(0, n):
        model = train_adaBoost(trees, lr, depth
                            ,train_data)
        
        if n == 1:
            return model
        
        y = sim(model, test_data, test_num)
        score = eval_prediction(y, test_data[test_num][:, -1], 0)
        
        if score < best_score:
            best_score = score
            best_model = model
        
    return best_model


def train_multi_adaBoost(trees, lr, depth
                      ,n, train_data, test_data, test_num):
    params=[]
    params.append(trees)
    params.append(lr)
    
    best_score = 1
    import itertools
    for par_el in itertools.product(*params):
        print(f'AdaBoost training parameters:')
        print(f'Number of estimators: {par_el[0]}')
        print(f'Learning rate: {par_el[1]}')
        
        model = train_adaBoost_nTimes(par_el[0], par_el[1], depth
                                   ,n,train_data, test_data, test_num)
        
        y = sim(model, test_data, test_num)
        score = eval_prediction(y, test_data[test_num][:,-1], 0)
        
        if score < best_score:
            best_score = score
            best_model = model
            
    return best_model


def train_NNadaBoost(trees, lr, depth
                ,train_data):
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.neural_network import MLPRegressor
    estimator = DecisionTreeRegressor(
        max_depth=depth
        ,max_features=1
    )
    
    model = AdaBoostRegressor(
        base_estimator=estimator
        ,n_estimators=trees
        ,learning_rate=lr
        ,loss='square'
    )
    
    model.fit(train_data[:,:-1], train_data[:,-1])
    return model


def train_NNadaBoost_nTimes(trees, lr, depth, n
                       ,train_data, test_data, test_num):
    best_score = 1
    for i in range(0, n):
        model = train_adaBoost(trees, lr, depth
                            ,train_data)
        
        if n == 1:
            return model
        
        y = sim(model, test_data, test_num)
        score = eval_prediction(y, test_data[test_num][:, -1], 0)
        
        if score < best_score:
            best_score = score
            best_model = model
        
    return best_model


def train_multi_NNadaBoost(trees, lr, depth
                      ,n, train_data, test_data, test_num):
    params=[]
    params.append(trees)
    params.append(lr)
    
    best_score = 1
    import itertools
    for par_el in itertools.product(*params):
        print(f'AdaBoost training parameters:')
        print(f'Number of estimators: {par_el[0]}')
        print(f'Learning rate: {par_el[1]}')
        
        model = train_adaBoost_nTimes(par_el[0], par_el[1], depth
                                   ,n,train_data, test_data, test_num)
        
        y = sim(model, test_data, test_num)
        score = eval_prediction(y, test_data[test_num][:,-1], 0)
        
        if score < best_score:
            best_score = score
            best_model = model
            
    return best_model


def create_dataset(X, y, time_steps=1):
    #based on tutorial:
    #https://towardsdatascience.com/
    #time-series-forecasting-with-lstms-using-tensorflow-2-and-keras-in-python-6ceee9c6c651
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


    
def prepare_timeseries(train_set, test_set_list, prepare):
    pd_X = pd.DataFrame(data=train_set[:, :-1])
    pd_Y = pd.DataFrame(data=train_set[:, -1])

    pd_TX = []
    pd_TY = []

    for test in test_set_list:
        pd_TX.append(pd.DataFrame(test[:, :-1]))
        pd_TY.append(pd.DataFrame(test[:, -1]))

    print(f'Sekwencjonuje zbior uczacy')
    Xs, Ys = create_dataset(pd_X, pd_Y, prepare)

    TXs = []
    TYs = []
    
    for i in range(0, len(pd_TX)):
        print(f'Sekswencjonuje zbior testowy nr: {i}')
        txs, tys = create_dataset(pd_TX[i], pd_TY[i], prepare)
        TXs.append(txs)
        TYs.append(tys)
        
    return Xs, Ys, TXs, TYs


def prepare_surge(test_set_list, prepare):
    pd_TX = []
    pd_TY = []

    for test in test_set_list:
        pd_TX.append(pd.DataFrame(test[:, :-1]))
        pd_TY.append(pd.DataFrame(test[:, -1]))

    TXs = []
    TYs = []
    
    for i in range(0, len(pd_TX)):
        print(f'Sekswencjonuje zbior testowy nr: {i}')
        txs, tys = create_dataset(pd_TX[i], pd_TY[i], prepare)
        TXs.append(txs)
        TYs.append(tys)
        
    return TXs, TYs


def train_LSTM_ntimes(structure, patience
                     ,epochs, batch_size
                     ,dropouts, rec_dropouts, N, debug_log
                     ,train_X, train_Y, set_num
                     ,test_X, test_Y, criterion):
    best_score = 1
    nets_trained = 0
    
    if debug_log == 'yes':
        import sys
        bck_stdout = sys.stdout
        sys.stdout = open('log.txt', 'w')
    
    while nets_trained < N:
        model = train_LSTM(structure=structure, patience=3
                          ,epochs=epochs, batch_size=batch_size
                          ,dropouts=dropouts, rec_dropouts=rec_dropouts
                          ,train_X=Xs, train_Y=Ys)
        
        y = model.predict(test_X[set_num])
        
        if criterion == 'mse':
            score = eval_prediction(y, test_Y[set_num][:, -1], 0)
            
        if criterion == 'max':
            residuals = np.copy(y[:, 0])
            for j in range(0, len(residuals)):
                residuals[j] = residuals[j] - \
                test_data[set_num][j, -1]
            score = np.max(np.abs(residuals))
            
        print(f"Wynik sieci nr {nets_trained+1}: {score}")
        
        if score < best_score:
            best_score = score
            best_model = model
        
        del model
        nets_trained = nets_trained + 1
    
    import winsound
    filename = 'sound.wav'
    winsound.PlaySound(filename, winsound.SND_FILENAME)
    
    if debug_log == 'yes':
        sys.stdout = bck_stdout
    return best_model


def train_LSTM(structure, patience
              ,epochs, batch_size
              ,dropouts, rec_dropouts
              ,train_X, train_Y):
    from keras.models import Sequential
    from keras.layers import Dense, Activation, LSTM, Dropout
    from keras.layers import Activation
    from keras.layers import LSTM
    from keras.layers import Masking
    from keras.callbacks import EarlyStopping    
    
    model = Sequential()
    for i in range(0, len(structure)):
        if i == 0:
            model.add(Masking(mask_value=-10
                             ,input_shape=(train_X.shape[1], train_X.shape[2])))
            
            model.add(LSTM(units=structure[i]
                          ,activation='tanh'
                          ,recurrent_activation='sigmoid'
                          ,dropout=dropouts[i]
                          ,recurrent_dropout=rec_dropouts[i]
                          ,return_sequences=False))
        else:
            model.add(Dense(units=structure[i]
                           ,activation='tanh'))
            model.add(Dropout(dropouts[i]))
                      
    model.add(Dense(units=1, activation='linear'))
    model.compile(loss='mean_squared_error'
                  ,optimizer='adam')

    es = EarlyStopping(monitor='val_loss', mode='min'
                      ,verbose=1, patience=patience)
    
    model.fit(train_X, train_Y, epochs=epochs,
              batch_size=batch_size,
              validation_split=0.35,
              verbose=1,
              shuffle=True,
              callbacks=[es])
    
    return model

def multi_train_LSTM(structure_list, patience
                    ,dropouts, rec_dropouts
                    ,epochs, batch_size, train_single
                    ,train_X, train_Y, debug_log
                    ,test_X, test_Y, set_num
                    ,criterion):
    
    from sklearn.metrics import mean_squared_error as mse
    
    if debug_log == 'yes':
        import sys
        bck_stdout = sys.stdout
        sys.stdout = open('log.txt', 'w')
    
    best_score = 1
    
    for i in range(0, len(structure_list)):
        print(f'Trenuje siec o strukturze: {structure_list[i]}')
        curr_model = train_LSTM_ntimes(
            structure=structure_list[i], patience=patience
            ,epochs=epochs, batch_size=batch_size
            ,rec_dropouts=rec_dropouts[i]
            ,dropouts=dropouts[i], N=train_single, debug_log='no'
            ,train_X=train_X, train_Y=train_Y, set_num=set_num
            ,test_X=test_X, test_Y=test_Y, criterion='mse')
        
        pred = curr_model.predict(test_X[set_num])
        curr_score = mse(pred[:, 0], test_Y[set_num])
        print(f'Wynik biezacej sieci: {curr_score}')
        
        if curr_score < best_score:
            best_score = curr_score
            best_model = curr_model
            
    import winsound
    filename = 'jasny chuj.wav'
    winsound.PlaySound(filename, winsound.SND_FILENAME)
    
    if debug_log == 'yes':
        sys.stdout = bck_stdout
    
    return best_model

def generate_SVR_params(C_min, C_max, C_step
                       ,eps_min, eps_max, eps_step):
    C_max = C_max + C_step
    C_vals = np.arange(C_min, C_max, C_step)
    
    eps_max = eps_max + eps_step
    eps_vals = np.arange(eps_min, eps_max, eps_step)
    
    params_list = []
    for C in C_vals:
        for eps in eps_vals:
            params_list.append([C, eps])
    
    return params_list

def try_add_column(cols_taken, test_col, criterion, train_set
                  ,test_set, model_type):
    from sklearn.metrics import mean_squared_error as mse
    col = len(train_set[0])
    best_score = 1
    best_col = 0

    for i in range(0, col):
        if i not in cols_taken:
            cols_test = [i]
            for _col in cols_taken:
                cols_test.append(_col)

            train_set_trim = train_set[:, cols_test]
            test_set_trim = []
            for test in test_set:
                test_set_trim.append(test[:, cols_test])
            
            if model_type == 'SVR':
                model = train_SVR(C=0.4, epsilon=0.014
                                  ,train_data=train_set_trim)
            
            pred = sim(model, test_set_trim, test_col)
            
            if criterion == 'mse':
                score = mse(pred[:, 0], test_set_trim[test_col][:, -1])
            if criterion == 'max':
                residuals = np.copy(pred[:, 0])
                for j in range(0, len(residuals)):
                    residuals[j] = residuals[j] - \
                    test_set_trim[test_col][j, -1]
                score = np.max(np.abs(residuals))

            if score < best_score:
                best_model = model
                best_score = score
                best_col = i

            print(cols_test)
            print(f'Test kolumny {i}: wynik: {score}')

    print(f'Najlepsza kolumna to: {best_col} z wynikiem: {best_score}')
    
def try_drop_column(start_cols, test_col, criterion
                   ,train_set ,test_set, model_type):
    
    from sklearn.metrics import mean_squared_error as mse
    best_score = 1
    best_col = 0

    for i in range(0, len(base_cols)):
        cols_test = np.copy(base_cols)
        cols_test = np.delete(base_cols, i)

        train_set_trim = train_set[:, cols_test]
        test_set_trim = []
        for test in test_set:
            test_set_trim.append(test[:, cols_test])
        
        if model_type == 'SVR':
            model = train_SVR(C=0.4, epsilon=0.014
                              ,train_data=train_set_trim)
        
        pred = sim(model, test_set_trim, test_col)
        
        if criterion == 'mse':
            score = mse(pred[:, 0], test_set_trim[test_col][:, -1])
        if criterion == 'max':
            residuals = np.copy(pred[:, 0])
            for j in range(0, len(residuals)):
                residuals[j] = residuals[j] - \
                test_set_trim[test_col][j, -1]
            score = np.max(np.abs(residuals))

        if score < best_score:
            best_model = model
            best_score = score
            best_col = i

        print(cols_test)
        print(f'Test kolumny {base_cols[i]}: wynik: {score}')

    print(f'Najlepsza kolumna to: {best_col} z wynikiem: {best_score}')

def dump_timeseries(Xs, Ys, TXs, TYs, seq_len):
    with open(f'ts_trim_train_in_{seq_len}.pkl', 'wb') as f:
        dill.dump(Xs,f)

    with open(f'ts_trim_train_out_{seq_len}.pkl', 'wb') as f:
        dill.dump(Ys,f)
        
    with open(f'ts_trim_test_in_{seq_len}.pkl', 'wb') as f:
        dill.dump(TXs,f)

    with open(f'ts_trim_test_out_{seq_len}.pkl', 'wb') as f:
        dill.dump(TYs,f)
        
        
def dump_surges(SXs, SYs, seq_len):
    with open(f'ts_surge_in_{seq_len}.pkl', 'wb') as f:
        dill.dump(SXs,f)

    with open(f'ts_surge_out_{seq_len}.pkl', 'wb') as f:
        dill.dump(SYs,f)
        
        
def load_timeseries(seq_len):
    with open(f'ts_trim_train_in_{seq_len}.pkl', 'rb') as f:
        Xs = dill.load(f)
        
    with open(f'ts_trim_train_out_{seq_len}.pkl', 'rb') as f:
        Ys = dill.load(f)
        
    with open(f'ts_trim_test_in_{seq_len}.pkl', 'rb') as f:
        TXs = dill.load(f)
        
    with open(f'ts_trim_test_out_{seq_len}.pkl', 'rb') as f:
        TYs = dill.load(f)
        
    return Xs, Ys, TXs, TYs


def load_surges(seq_len):
    with open(f'ts_surge_in_{seq_len}.pkl', 'rb') as f:
        SXs = dill.load(f)

    with open(f'ts_surge_out_{seq_len}.pkl', 'rb') as f:
        SYs = dill.load(f)
    
    return SXs, SYs


import sys
bckStdout = sys.stdout