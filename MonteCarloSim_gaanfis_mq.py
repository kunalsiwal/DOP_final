# %%
counts={}
for i in range(10):
   # %%
    from sklearn.model_selection import train_test_split 
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    data = pd.read_csv('CONVERTED_DOP_DATA.csv')
    data

    # %%
    #calculating for MS
    X=data.iloc[:,0:4]
    X

    # %%
    Y=data.iloc[:,6]
    Y

    # %%
    bins=[0,0.2,0.3,0.5]
    aa=X['Cellulose Fibers(%)']
    binned=np.digitize(aa,bins)
    binned


    # %%
    X_train , X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,
                                            random_state=101)
    X_train.shape,Y_train.shape

    # %%
    X_train


    # %%
    scaler_x=MinMaxScaler()
    scaler_y=MinMaxScaler()
    scaler_x.fit(X)
    X_train=scaler_x.transform(X_train)
    X_test=scaler_x.transform(X_test)
    # Y_train=Y_train
    # Y_test=Y_test

    Y_train = (Y_train-np.min(Y_train))/(np.max(Y_train)-np.min(Y_train))
    Y_test= (Y_test-np.min(Y_test))/(np.max(Y_test)-np.min(Y_test))


    # %%
    # from ANFIS import EVOLUTIONARY_ANFIS
    # E_Anfis = EVOLUTIONARY_ANFIS(functions=3,generations=500,offsprings=10,
    #                              mutationRate=0.7,learningRate=0.4,chance=0.7,ruleComb="simple")
    from personalANFIS import GAANFIS
    gaanfis=GAANFIS(n_input=X.shape[1], n_mf=3, n_output=1, population_size=25, max_generations=100, mutation_rate=0.9)
    gaanfis.train(X_train, Y_train)
    Y_pred=gaanfis.predict(X_test)
    from githubANFIS import EVOLUTIONARY_ANFIS
    E_Anfis=EVOLUTIONARY_ANFIS(functions=3,generations=100,offsprings=25,mutationRate=0.7,learningRate=0.4,chance=0.7,ruleComb="simple")
    bestParam,bestModel =E_Anfis.fit(X_train,np.array(Y_train).reshape(-1,1))
    Y_pred_git=E_Anfis.predict(X_test,bestParam,bestModel)
    from sklearn.metrics import mean_squared_error
    mse=mean_squared_error(Y_test,Y_pred)
    mse2=mean_squared_error(Y_test,np.ravel(Y_pred_git))
    mse,mse2


    # %%
    from sklearn.metrics import r2_score
    score=r2_score(Y_test,Y_pred)
    score_=r2_score(Y_test,np.ravel(Y_pred_git))
    score,score_


    # %%
    from sklearn.metrics import mean_squared_error
    score2=mean_squared_error(Y_test,Y_pred,squared=False)
    score2_=mean_squared_error(Y_test,np.ravel(Y_pred_git),squared=False)
    score2,score2_

    # %%
    from sklearn.metrics import mean_absolute_error
    score3=mean_absolute_error(Y_test,Y_pred)
    score3_=mean_absolute_error(Y_test,np.ravel(Y_pred_git))
    score3,score3_

    # %%

    from scipy import stats
    res = stats.pearsonr(Y_test,Y_pred)
    res_=stats.pearsonr(Y_test,np.ravel(Y_pred_git))
    res,res_

    # %%
    # plt.scatter(Y_test,np.ravel(Y_pred_git))
    # plt.xlabel('Actual value')
    # plt.ylabel('Predicted Value')
    # plt.show()

    # %%
    Y_test

    # %%
    np.ravel(Y_pred_git)
    rounded_res=round(res_.statistic,4)
    if rounded_res in counts:
        counts[rounded_res]=counts[rounded_res]+1
    else:
        counts[rounded_res]=0
    print(counts)
    
print(counts)
# %%
# plt.scatter(Y_test,Y_pred)
# plt.xlabel('Actual value')
# plt.ylabel('Predicted Value')
# plt.show()

# %%



