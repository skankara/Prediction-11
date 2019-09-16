import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import operator


testset = pd.read_csv('ipl/testingdata.csv')
trainset = pd.read_csv('ipl/trainingdata.csv')

testset = testset.loc[testset.match_id == 7894,:]

x_cols = ['inning', 'over', 'total_runs', 'player_dismissed', 'innings_wickets', 'innings_score', 'aim', 'remaining_runs', 'run_rate', 'required_run_rate', 'rr_diff', 'is_batting_team']

#testset = trainingdata.loc[trainingdata.match_id == 5,:]
#trainset = trainingdata.loc[trainingdata.match_id != 5,:]

team1 = testset.team1
team2 = testset.team2
team1 = team1.iloc[0]
team2 = team2.iloc[0]

# create the input and target variables #
trainset_x = np.array(trainset[x_cols[:]])
trainset_y = np.array(trainset['target'])

testset_x = np.array(testset[x_cols[:]])[:-1,:]
testset_y = np.array(testset['target'])[:-1]
print("Training data : ",trainset_x.shape, trainset_y.shape)
print("Testing data : ",testset_x.shape, testset_y.shape)

param = {}
param['objective'] = 'binary:logistic'
param['eta'] = 0.05
param['max_depth'] = 8
param['silent'] = 1
param['eval_metric'] = "auc"
param['min_child_weight'] = 1
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['seed'] = 0
num_rounds = 100
plst = list(param.items())
xgtrain = xgb.DMatrix(trainset_x, label=trainset_y)
model = xgb.train(plst, xgtrain, num_rounds)
testing = xgb.DMatrix(testset_x)
prediction = model.predict(testing)

df = pd.DataFrame(prediction)
df.to_csv('ipl/prediction.csv',index=False)


outputdata = pd.DataFrame({'Team1':testset.team1.values})
outputdata['is_batting_team'] = testset.is_batting_team.values
outputdata['innings_over'] = np.array(testset.apply(lambda row: str(row['inning']) + "_" + str(row['over']), axis=1))
outputdata['innings_score'] = testset.innings_score.values
outputdata['innings_wickets'] = testset.innings_wickets.values
outputdata['aim'] = testset.aim.values
outputdata['total_runs'] = testset.total_runs.values
outputdata['predictions'] = list(prediction)+[1]

fig, axX = plt.subplots(figsize=(12,6))
axY = axX.twinx()
labels = np.array(outputdata['innings_over'])
ind = np.arange(len(labels))
width = 0.7
rects = axX.bar(ind, np.array(outputdata['innings_score']), width=width, color=['orange']*20 + ['blue']*20)
axX.set_xticks(ind+((width)/2.))
axX.set_xticklabels(labels, rotation='vertical')
axX.set_ylabel("Innings score")
axX.set_xlabel("Innings and over")
title = "Win percentage prediction for " + str(team1) +" Vs "+ str(team2)
axX.set_title(str(title))
#print("Winner : ", testset['winner'])
#print(testset['winner'].unique())
if((testset['winner'].unique()) == team1):
    axY.plot(ind+0.35, np.array(outputdata['predictions']), color='red', marker='o')
else :
    axY.plot(ind + 0.35, np.array(outputdata['predictions']), color='cyan', marker='o')
#axY.plot(ind+0.35, np.array([0.5]*40), color='red', marker='o')
axY.set_ylabel("Win percentage", color='g')
axY.set_ylim([0,1])
axY.grid(b=False)
plt.show()

fig, ax1 = plt.subplots(figsize=(12,6))
ax2 = ax1.twinx()
labels = np.array(outputdata['innings_over'])
ind = np.arange(len(labels))
width = 0.7
rects = ax1.bar(ind, np.array(outputdata['total_runs']), width=width, color=['orange']*20 + ['blue']*20)
ax1.set_xticks(ind+((width)/2.))
ax1.set_xticklabels(labels, rotation='vertical')
ax1.set_ylabel("Runs in the given over")
ax1.set_xlabel("Innings and over")
title = "Win percentage prediction for (each over) "+ team1 +" Vs "+ team2
ax1.set_title(str(title))
if((testset['winner'].unique()) == team1):
    ax2.plot(ind+0.35, np.array(outputdata['predictions']), color='red', marker='o')
else:
    ax2.plot(ind + 0.35, np.array(outputdata['predictions']), color='cyan', marker='o')
#ax2.plot(ind+0.35, np.array(outputdata['predictions']), color='green', marker='o')
#ax2.plot(ind+0.35, np.array([0.5]*40), color='red', marker='o')
ax2.set_ylabel("Win percentage", color='green')
ax2.set_ylim([0,1])
ax2.grid(b=False)
plt.show()


