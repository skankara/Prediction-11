import pandas as pd
import json
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from collections import Counter

def most_frequent(preds):
    cnt = Counter(preds)
    #print(cnt)
    #print("cnt :",cnt.most_common(1))
    #return cnt.most_common()
    return sorted(cnt, key=cnt.get, reverse=True)


def winningProbabolity(team1, team2, city, toss_winner):
    home = ""
    if city == "Hyderabad" or city =="Visakhapatnam":
        home = "SRH"
    if city == "Pune":
        home = "Pune"
    if city == "Rajkot" or city == "Ahmedabad":
        home = "GL"
    if city == "Bangalore":
        home = "RCB"
    if city =="Mumbai" or city =="Nagpur":
        home = "MI"
    if city =="Kolkata" or city == "Cuttack":
        home = "KKR"
    if city =="Delhi" or city =="Raipur":
        home = "DD"
    if city == "Chandigarh" or city =="Dharamsala" or city == "Indore" or city == "Mohali":
        home = "KXIP"
    if city =="Jaipur":
        home = "RR"
    if city =="Chennai" :
        home = "CSK"
    if city == "Ranchi" :
        if team1 == "CSK" or team2 == "CSK":
            home = "CSK"
        else:
            home ="KKR"

    if home == team1 or home == team2:
        return home
    else:
        return toss_winner




def predict(team1, team2, city, toss_winner, toss_decision):
#def predict(input):
    with open('ipl/teamCodes.json', encoding='utf-8') as data_file:
        teams = json.loads(data_file.read())
    data_file.close()
    #print(teams)

    with open('ipl/venueCodes.json', encoding='utf-8') as data_file:
        venue = json.loads(data_file.read())
    #print(venue)
    data_file.close()

    with open('ipl/tossCodes.json', encoding='utf-8') as data_file:
        toss = json.loads(data_file.read())
    #print(toss)
    data_file.close()

    with open('ipl/reverseteamCodes.json', encoding='utf-8') as data_file:
        reverseteams = json.loads(data_file.read())
    #print(reverseteams)
    data_file.close()
    print("Input : ")
    print("Team1 : ",team1)
    print("Team2 : ",team2)
    print("City : ", city)
    print("Toss Winner : ", toss_winner)
    print("Toss Decision : ", toss_decision)
    # print(homeTeam, awayTeam, City, tossW, tossD)
    input=[]
    input.append(teams[team1])
    input.append(teams[team2])
    input.append(venue[city])
    input.append(teams[toss_winner])
    input.append(toss[toss_decision])

    #print("Numerical Input :", input)


    matches_data = pd.read_csv('ipl/matches_new.csv')
    matches_data = matches_data[['season','team1', 'team2', 'city', 'toss_winner', 'toss_decision', 'winner']]

    training = matches_data.loc[matches_data.season != 2018]
    testing = matches_data.loc[matches_data.season == 2018]
    training = training[['team1', 'team2', 'city', 'toss_winner', 'toss_decision', 'winner']]
    testing = testing[['team1', 'team2', 'city', 'toss_winner', 'toss_decision', 'winner']]
    testing.to_csv('ipl/team_prediction.csv',index=False)


    trainvector = training.values
    x_train = trainvector[:, 0:5]
    y_train = trainvector[:, 5]
    testvector = testing.values
    x_test = testvector[:, 0:5]
    y_test = testvector[:, 5]


    predictions =[]
    model1 = DecisionTreeClassifier(random_state=1)
    model1.fit(x_train,y_train)
    model11 = DecisionTreeClassifier(criterion="entropy",random_state=1)
    model11.fit(x_train, y_train)

    model2 = RandomForestClassifier(n_estimators=10)
    model2.fit(x_train, y_train)

    model3 = MLPClassifier(hidden_layer_sizes=(3,), activation='logistic',
                       solver='lbfgs', alpha=0.0001,learning_rate='constant',
                      learning_rate_init=0.001, max_iter= 10000)
    model3.fit(x_train, y_train)
    model4 = SVC(gamma='auto', probability=True)
    model4.fit(x_train,y_train)
    model6 = KNeighborsClassifier()
    model6.fit(x_train,y_train)
    pred1 = model1.predict([input])
    accu1 = model1.predict(x_test)
    pred11 = model11.predict([input])
    accu11 = model11.predict(x_test)
    pred2 = model2.predict([input])
    accu2 = model2.predict(x_test)
    pred3 = model3.predict([input])
    accu3 = model3.predict(x_test)
    pred4 = model4.predict([input])
    accu4 = model4.predict(x_test)
    model5 = LogisticRegression(multi_class='auto',solver='lbfgs',max_iter=10000).fit(x_train, y_train)
    pred5 = model5.predict([input])
    accu5 = model5.predict(x_test)
    pred6 = model6.predict([input])
    accu6 = model6.predict(x_test)

    model17 = GaussianNB()
    model17.fit(x_train, y_train)
    model18 = LinearSVC(max_iter=100000)
    model18.fit(x_train, y_train)

    pred17 = model17.predict([input])
    accu17 = model17.predict(x_test)
    pred18 = model18.predict([input])
    accu18 = model18.predict(x_test)

    predictions.append(reverseteams[str(pred1[0])])
    predictions.append(reverseteams[str(pred2[0])])
    predictions.append(reverseteams[str(pred3[0])])
    predictions.append(reverseteams[str(pred4[0])])
    #predictions.append(reverseteams[str(pred5[0])])
    predictions.append(reverseteams[str(pred6[0])])


    #predictions.append(reverseteams[str(pred17[0])])
    #predictions.append(reverseteams[str(pred18[0])])

    print("<30% accuracy : ")
    print("Gaussian Naive Bayes :", reverseteams[str(pred17[0])])
    print("Gaussian Naive Bayes  Accuracy : ", round(accuracy_score(y_test, accu17)*100,2))
    print("Linear SVC :", reverseteams[str(pred18[0])])
    print("Linear SVC Accuracy : ", round(accuracy_score(y_test, accu18)*100,2))
    #print("Logistic Regression :", reverseteams[str(pred5[0])])
    #print("Logistic Regression Accuracy : ", round(accuracy_score(y_test, accu5) * 100, 2))

    print("\n>30% accuracy : ")
    print("DecisionTreeClassifier :", reverseteams[str(pred1[0])])
    print("DecisionTreeClassifier Accuracy : ", round(accuracy_score(y_test, accu1)*100,2))
    print("DecisionTreeClassifier with entropy:", reverseteams[str(pred11[0])])
    print("DecisionTreeClassifier Accuracy : ", round(accuracy_score(y_test, accu11) * 100, 2))
    print("SVC :", reverseteams[str(pred4[0])])
    print("SVC Accuracy : ", round(accuracy_score(y_test, accu4)*100,2))
    print("KNeighbors Classifier :", reverseteams[str(pred6[0])])
    print("KNeighbors Classifier Accuracy : ", round(accuracy_score(y_test, accu6)*100,2))
    print("RandomForestClassifier :", reverseteams[str(pred2[0])])
    print("RandomForestClassifier Accuracy : ", round(accuracy_score(y_test, accu2)*100,2))
    print("MLPClassifier  :", reverseteams[str(pred3[0])])
    print("MLPClassifier  Accuracy : ", round(accuracy_score(y_test, accu3)*100,2))


    #Bagging
    #Building multiple models(same type) from different subsamples of the training dataset
    seed = 7
    kfold = model_selection.KFold(n_splits=10, random_state= seed)
    #cart = DecisionTreeClassifier()
    num_trees = 100
    model8 = BaggingClassifier(base_estimator=model1, n_estimators=num_trees, random_state=seed)
    # results = model_selection.cross_val_predict(model,x,y, cv=kfold)
    # print(results.mean())
    model8.fit(x_train,y_train)
    pred8 = model8.predict([input])
    predictions.append(reverseteams[str(pred8[0])])
    print("Bagging Prediction : ",reverseteams[str(pred8[0])])
    print("Bagging Accuracy : ", round((model8.score(x_test,y_test))*100,2))
    #print(results)

    #Boosting
    num_trees = 30
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    #model9 = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
    model9 = GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10)
    model9.fit(x_train,y_train)
    pred9 = model9.predict([input])
    predictions.append(reverseteams[str(pred9[0])])
    print("GradientBoostingClassifier(Adaboost) Prediction : ",reverseteams[str(pred9[0])])
    print("GradientBoostingClassifier(Adaboost) Accuracy : ",round((model9.score(x_test,y_test))*100,2))

    model7 = VotingClassifier(estimators=[('bg',model8),('bo',model9), ('dt', model1), ('rf', model2),('ls', model3), ('sv', model4),('kn', model6)], voting='soft')
    model7.fit(x_train, y_train)
    pred7 = model7.predict([input])
    print("Voting Prediction ", reverseteams[str(pred7[0])])
    print("Voting Prediction Accuracy : ",round((model7.score(x_test, y_test))*100,2))
    predictions.append(reverseteams[str(pred7[0])])

    frequent = most_frequent(predictions)
    #print(type(frequent))
    #print(type(team1))
    #print(type(team2))
    #print(frequent)
    for strwinner in frequent:
        if strwinner == team1 or strwinner == team2:
            #print("winner",strwinner)
            return strwinner
    return winningProbabolity(team1,team2,city,toss_winner)