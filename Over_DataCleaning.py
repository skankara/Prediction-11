import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import operator
def get_remaining_runs(dr):
    if dr['aim'] == -1.:
        return -1
    else:
        return dr['aim'] - dr['innings_score']

def get_required_runrate(dr):
    if dr['remaining_runs'] == -1:
        return -1.
    elif dr['over'] == 20:
        return 99
    else:
        return dr['remaining_runs'] / (20-dr['over'])

def get_rr_diff(dr):
    if dr['inning'] == 1:
        return -1
    else:
        return dr['run_rate'] - dr['required_run_rate']

def labeling(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,'%f' % float(height), ha='center', va='bottom')


matches = pd.read_csv('ipl/matches.csv')
deliveries = pd.read_csv('ipl/deliveries.csv')

matches = matches.loc[matches.season!=2018,:]
matches = matches.loc[matches.dl_applied == 0,:]
#print(matches.head())

matches.to_csv('ipl/trail.csv',index=False)
scoredata = pd.merge(deliveries,matches[['id','season','winner','result','dl_applied','team1','team2']],left_on='match_id',right_on='id')
scoredata.player_dismissed.fillna(0, inplace = True) #it will replace NaN values with 0
scoredata.loc[scoredata['player_dismissed'] != 0, 'player_dismissed'] =1
#scoredata.to_csv('ipl/scoredata1.csv',index=False)
#print(scoredata)
trainingdata = scoredata.groupby(['match_id', 'inning', 'over', 'team1', 'team2', 'batting_team', 'winner'])[['total_runs', 'player_dismissed']].agg(['sum']).reset_index()
trainingdata.columns = trainingdata.columns.get_level_values(0)
#trainingdata.to_csv('ipl/trainingdata.csv',index=False)
#match_id,	inning,	over,	team1,	team2,	batting_team,	winner,	total_runs,	player_dismissed

trainingdata['innings_wickets'] = trainingdata.groupby(['match_id', 'inning'])['player_dismissed'].cumsum()
trainingdata['innings_score'] = trainingdata.groupby(['match_id', 'inning'])['total_runs'].cumsum()

runs_data = trainingdata.groupby(['match_id', 'inning'])['total_runs'].sum().reset_index()
runs_data = runs_data.loc[runs_data['inning']==1,:]
runs_data['inning'] = 2
#aim = score_target
runs_data.columns = ['match_id', 'inning', 'aim']
trainingdata = trainingdata.merge(runs_data, how='left', on = ['match_id', 'inning'])
trainingdata['aim'].fillna(-1, inplace=True)
##scoredata.player_dismissed.fillna(0, inplace = True)
#
# #remaining_runs = remaining_target
trainingdata['remaining_runs'] = trainingdata.apply(lambda row: get_remaining_runs(row),axis=1)
#
# # get the run rate #
trainingdata['run_rate'] = trainingdata['innings_score'] / trainingdata['over']
#
trainingdata['required_run_rate'] = trainingdata.apply(lambda row: get_required_runrate(row), axis=1)

trainingdata['rr_diff'] = trainingdata.apply(lambda row: get_rr_diff(row), axis=1)
trainingdata['is_batting_team'] = (trainingdata['team1'] == trainingdata['batting_team']).astype('int')
trainingdata['target'] = (trainingdata['team1'] == trainingdata['winner']).astype('int')
trainingdata.to_csv('ipl/trainingdata.csv',index=False)

#print(trainingdata.head())

#x_cols = ['inning', 'over', 'total_runs', 'player_dismissed', 'innings_wickets', 'innings_score', 'aim', 'remaining_runs', 'run_rate', 'required_run_rate', 'rr_diff', 'is_batting_team']

matches = pd.read_csv('ipl/matches.csv')
deliveries = pd.read_csv('ipl/deliveries.csv')

matches = matches.loc[matches.season==2018,:]
matches = matches.loc[matches.dl_applied == 0,:]
#print(matches.head())

matches.to_csv('ipl/trail.csv',index=False)
scoredata = pd.merge(deliveries,matches[['id','season','winner','result','dl_applied','team1','team2']],left_on='match_id',right_on='id')
scoredata.player_dismissed.fillna(0, inplace = True) #it will replace NaN values with 0
scoredata.loc[scoredata['player_dismissed'] != 0, 'player_dismissed'] =1
#scoredata.to_csv('ipl/scoredata1.csv',index=False)
#print(scoredata)
testingdata = scoredata.groupby(['match_id', 'inning', 'over', 'team1', 'team2', 'batting_team', 'winner'])[['total_runs', 'player_dismissed']].agg(['sum']).reset_index()
testingdata.columns = testingdata.columns.get_level_values(0)
#trainingdata.to_csv('ipl/trainingdata.csv',index=False)
#match_id,	inning,	over,	team1,	team2,	batting_team,	winner,	total_runs,	player_dismissed

testingdata['innings_wickets'] = testingdata.groupby(['match_id', 'inning'])['player_dismissed'].cumsum()
testingdata['innings_score'] = testingdata.groupby(['match_id', 'inning'])['total_runs'].cumsum()

runs_data = testingdata.groupby(['match_id', 'inning'])['total_runs'].sum().reset_index()
runs_data = runs_data.loc[runs_data['inning']==1,:]
runs_data['inning'] = 2
#aim = score_target
runs_data.columns = ['match_id', 'inning', 'aim']
testingdata = testingdata.merge(runs_data, how='left', on = ['match_id', 'inning'])
testingdata['aim'].fillna(-1, inplace=True)
##scoredata.player_dismissed.fillna(0, inplace = True)
#
# #remaining_runs = remaining_target
testingdata['remaining_runs'] = testingdata.apply(lambda row: get_remaining_runs(row),axis=1)
#
# # get the run rate #
testingdata['run_rate'] = testingdata['innings_score'] / testingdata['over']
#
testingdata['required_run_rate'] = testingdata.apply(lambda row: get_required_runrate(row), axis=1)

testingdata['rr_diff'] = testingdata.apply(lambda row: get_rr_diff(row), axis=1)
testingdata['is_batting_team'] = (testingdata['team1'] == testingdata['batting_team']).astype('int')
testingdata['target'] = (testingdata['team1'] == testingdata['winner']).astype('int')
testingdata.to_csv('ipl/testingdata.csv',index=False)
