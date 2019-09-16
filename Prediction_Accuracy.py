import TeamVsTeamPrediction as tp
import pandas as pd
correct = 0
incorrect = 0

data_2018 = pd.read_csv('ipl/team_prediction.csv')

data_2018 = pd.read_csv('ipl/matches_codes1.csv')
data_2018.replace('Mumbai Indians', 'MI',inplace = True)
data_2018.replace('Delhi Daredevils', 'DD',inplace = True)
data_2018.replace('Royal Challengers Bangalore', 'RCB',inplace = True)
data_2018.replace('Rajasthan Royals', 'RR',inplace = True)
data_2018.replace('Kolkata Knight Riders', 'KKR',inplace = True)
data_2018.replace('Kings XI Punjab', 'KXIP',inplace = True)
data_2018.replace('Chennai Super Kings', 'CSK',inplace = True)
data_2018.replace('Sunrisers Hyderabad', 'SRH',inplace = True)

data_2018['prediction'] = ""

for i in range(len(data_2018)):
    print("i",i)
    input = []
    team1 = data_2018.team1[i]
    team2 = data_2018.team2[i]
    city = data_2018.city[i]
    toss_winner = data_2018.toss_winner[i]
    toss_decision = data_2018.toss_decision[i]
    won = tp.predict(team1, team2, city, toss_winner, toss_decision)
    print("winner : ",won)
    data_2018['prediction'][i] = won

data_2018.to_csv("ipl/team_vs_team_prediction_2018.csv")

data_2018 = pd.read_csv('ipl/team_vs_team_prediction_2018.csv')
for i in range(len(data_2018)):
    winner = data_2018.winner[i]
    prediction = data_2018.prediction[i]
    if winner == prediction :
        correct=correct+1
    else :
        incorrect = incorrect+1

total = correct+incorrect
accuracy = round((correct/total)*100,2)
print("Accuracy : ", accuracy)
