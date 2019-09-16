import TeamVsTeamPrediction as tp
import pandas as pd
correct = 0
incorrect = 0
data_2018 = pd.read_csv('ipl/team_vs_team_prediction_2018.csv')
for i in range(len(data_2018)):
    winner = data_2018.winner[i]
    prediction = data_2018.prediction[i]
    if winner == prediction :
        correct=correct+1
    else:
        incorrect = incorrect+1

total = correct+incorrect
accuracy = round((correct/total)*100,2)
#winner = tp.predict('DD', 'MI', 'Delhi', 'DD', 'bat') #DD
winner = tp.predict('DD', 'SRH', 'Delhi', 'DD', 'bat') #SRH
print("\nWinner : ",winner)
print("Accuracy : ", accuracy)

#winner = tp.predict('RR', 'KXIP', 'Jaipur', 'RR', 'bat') #RR
#print("\nWinner : ",winner)
#tp.predict('SRH', 'MI', 'Hyderabad', 'SRH', 'field') #SRH
# tp.predict('RR', 'SRH', 'Hyderabad', 'SRH', 'field') #SRH
#tp.predict('MI', 'CSK', 'Mumbai', 'CSK', 'field') #CSK
#winner = tp.predict('MI', 'RCB', 'Mumbai', 'RCB', 'field') #MI
#winner = tp.predict('DD', 'MI', 'Delhi', 'DD', 'bat') #DD
#winner = tp.predict('KXIP', 'RCB', 'Indore', 'RCB', 'field') #RCB
