import pandas as pd
import json
import csv
import SupportingFunctions as sf
pd.options.mode.chained_assignment = None  # default='warn'

teams = []
#teamsSet = {}
venue = []
toss = []
batsman = []
bowler = []
player = []

matches = pd.read_csv('ipl/matches.csv')
matches.drop('umpire1',axis=1, inplace=True)
matches.drop('umpire2',axis=1, inplace=True)
matches.drop('umpire3',axis=1, inplace=True)
print(matches.columns.unique())

matches.replace( 'Rising Pune Supergiant', 'Rising Pune Supergiants',inplace = True)
matches.replace( 'Pune Warriors', 'Rising Pune Supergiants',inplace = True)
matches.replace( 'Deccan Chargers', 'Sunrisers Hyderabad',inplace = True)
print(matches['team1'].unique())
matches['city'].fillna( matches['venue'].apply(lambda x: x[:5]),inplace = True)
matches.replace( 'Bengaluru', 'Bangalore',inplace = True)
print(matches['city'].unique())
print(matches['season'].unique())
matches.replace( 'Rising Pune Supergiant', 'Rising Pune Supergiants',inplace = True)

for i in range(0, len(matches)):
    teams.append(matches['team1'][i])
    venue.append(matches['city'][i])
    toss.append(matches['toss_decision'][i])

deliveries = pd.read_csv('ipl/deliveries.csv')
for i in range(0, len(deliveries)):
    batsman.append(deliveries['batsman'][i])
    bowler.append(deliveries['bowler'][i])
    player.append(deliveries['batsman'][i])
    player.append(deliveries['bowler'][i])

#
#
# #Create json files for further referrence
# with open('ipl/teamCodes.json','w') as outfile:
#     json.dump(teamNos, outfile)
teamSF =[]
for name in teams:
    if "Pune" in name :
        teamSF.append("Pune")
    if "Chennai" in name:
        teamSF.append("CSK")
    if "Punjab" in name:
        teamSF.append("KXIP")
    if "Hyderabad" in name or "Deccan" in name:
        teamSF.append("SRH")
    if "Mumbai" in name:
        teamSF.append("MI")
    if "Gujarat" in name:
        teamSF.append("GL")
    if "Bangalore" in name:
        teamSF.append("RCB")
    if "Kolkata" in name:
        teamSF.append("KKR")
    if "Delhi" in name:
        teamSF.append("DD")
    if "Rajasthan" in name:
        teamSF.append("RR")
    if "Kochi" in name:
        teamSF.append("Kochi")

teamNos,reverseteamNos = sf.assign_number(teamSF)
print(teamNos)
venueNos, reversevenueNos  = sf.assign_number(venue)
#print(venueNos)
tossNos, reversetossNos = sf.assign_number(toss)
#print(tossNos)
batsmanlist, batsmanidlist = sf.assign_number(batsman)
bowlerlist, bowleridlist = sf.assign_number(bowler)
playerlist, playeridlist = sf.assign_number(player)
with open('ipl/reverseteamCodes.json','w') as outfile:
    json.dump(reverseteamNos, outfile)
with open('ipl/tossCodes.json','w') as outfile:
    json.dump(tossNos, outfile)
with open('ipl/venueCodes.json','w') as outfile:
    json.dump(venueNos, outfile)
with open('ipl/batsman.json','w') as outfile:
    json.dump(batsmanlist, outfile)
with open('ipl/batsmanid.json','w') as outfile:
    json.dump(batsmanidlist, outfile)
with open('ipl/bowler.json','w') as outfile:
    json.dump(bowlerlist, outfile)
with open('ipl/bowlerid.json','w') as outfile:
    json.dump(bowleridlist, outfile)
with open('ipl/player.json','w') as outfile:
    json.dump(playerlist, outfile)
with open('ipl/playerid.json','w') as outfile:
    json.dump(playeridlist, outfile)

#creating new_matches.csv replaced with codes
#sf.replaceColumn(matches, teamNos)
for element in matches:
    for i in range(len(matches)):
        if element == 'team1' or element == 'team2' or element == 'toss_winner' or element == 'winner':
            #print(matches[element][i])
            #if (matches[element][i]) in teamNos:
            if "Pune" in str(matches[element][i]):
                key = "Pune"
            if "Chennai" in str(matches[element][i]):
                key="CSK"
            if "Punjab" in str(matches[element][i]):
                key="KXIP"
            if "Hyderabad" in str(matches[element][i]) or "Deccan" in str(matches[element][i]):
                key = "SRH"
            if "Mumbai" in str(matches[element][i]):
                key = "MI"
            if "Gujarat" in str(matches[element][i]):
                key = "GL"
            if "Bangalore" in str(matches[element][i]):
                key = "RCB"
            if "Kolkata" in str(matches[element][i]):
                key= "KKR"
            if "Delhi" in str(matches[element][i]):
                key = "DD"
            if "Rajasthan" in str(matches[element][i]):
                key = "RR"
            if "Kochi" in str(matches[element][i]):
                key = "Kochi"
                #key = matches[element][i]
            matches[element][i] = teamNos[key]
        if element == 'city':
            if (matches[element][i]) in venueNos:
                key = matches[element][i]
                matches[element][i] = venueNos[key]
        if element == 'toss_decision':
            if (matches[element][i]) in tossNos:
                key = matches[element][i]
                matches[element][i] = tossNos[key]
matches.to_csv('ipl/matches_new.csv',index=False)

# for element in deliveries:
#     for i in range(len(deliveries)):
#         if element == 'batting_team' or element == 'bowling_team':
#             #print(matches[element][i])
#             #if (matches[element][i]) in teamNos:
#             if "Pune" in str(deliveries[element][i]):
#                 key = "Pune"
#             if "Chennai" in str(deliveries[element][i]):
#                 key="CSK"
#             if "Punjab" in str(deliveries[element][i]):
#                 key="KXIP"
#             if "Hyderabad" in str(deliveries[element][i]) or "Deccan" in str(deliveries[element][i]):
#                 key = "SRH"
#             if "Mumbai" in str(deliveries[element][i]):
#                 key = "MI"
#             if "Gujarat" in str(deliveries[element][i]):
#                 key = "GL"
#             if "Bangalore" in str(deliveries[element][i]):
#                 key = "RCB"
#             if "Kolkata" in str(deliveries[element][i]):
#                 key= "KKR"
#             if "Delhi" in str(deliveries[element][i]):
#                 key = "DD"
#             if "Rajasthan" in str(deliveries[element][i]):
#                 key = "RR"
#             if "Kochi" in str(deliveries[element][i]):
#                 key = "Kochi"
#                 #key = matches[element][i]
#             deliveries[element][i] = teamNos[key]
#         if element == 'batsman' or element == 'non_striker' or element == 'bowler':
#             if (deliveries[element][i]) in playerlist:
#                 key = deliveries[element][i]
#                 deliveries[element][i] = playerlist[key]


#print(type(matches))
#matches_cleaned_data = pd.read_csv('ipl/matches_new.csv')
#deliveries.to_csv('ipl/deliveries_new.csv',index=False)

matches = pd.read_csv('ipl/matches.csv')
matches = matches.loc[matches.season==2018,:]
matches = matches[['city', 'team1','team2', 'toss_winner', 'toss_decision','winner']]
# matches.drop('umpire1',axis=1, inplace=True)
# matches.drop('umpire2',axis=1, inplace=True)
# matches.drop('umpire3',axis=1, inplace=True)
# print(matches.columns.unique())

matches.replace( 'Rising Pune Supergiant', 'Rising Pune Supergiants',inplace = True)
matches.replace( 'Pune Warriors', 'Rising Pune Supergiants',inplace = True)
matches.replace( 'Deccan Chargers', 'Sunrisers Hyderabad',inplace = True)
print(matches['team1'].unique())
#matches['city'].fillna( matches['venue'].apply(lambda x: x[:5]),inplace = True)
matches.replace( 'Bengaluru', 'Bangalore',inplace = True)
#print(matches['city'].unique())
#print(matches['season'].unique())
matches.replace( 'Rising Pune Supergiant', 'Rising Pune Supergiants',inplace = True)
matches.to_csv('ipl/matches_codes1.csv',index=False)

# for element in matches:
#     for i in range(1,len(matches)):
#         if element == 'team1' or element == 'team2' or element == 'toss_winner' or element == 'winner':
#             print(matches[element][i])
#             #if (matches[element][i]) in teamNos:
#             if "Pune" in str(matches[element][i]):
#                 key = "Pune"
#             if "Chennai" in str(matches[element][i]):
#                 key="CSK"
#             if "Punjab" in str(matches[element][i]):
#                 key="KXIP"
#             if "Hyderabad" in str(matches[element][i]) or "Deccan" in str(matches[element][i]):
#                 key = "SRH"
#             if "Mumbai" in str(matches[element][i]):
#                 key = "MI"
#             if "Gujarat" in str(matches[element][i]):
#                 key = "GL"
#             if "Bangalore" in str(matches[element][i]):
#                 key = "RCB"
#             if "Kolkata" in str(matches[element][i]):
#                 key= "KKR"
#             if "Delhi" in str(matches[element][i]):
#                 key = "DD"
#             if "Rajasthan" in str(matches[element][i]):
#                 key = "RR"
#             if "Kochi" in str(matches[element][i]):
#                 key = "Kochi"
#                 #key = matches[element][i]
#             matches[element][i] = key
#         # if element == 'city':
#         #     if (matches[element][i]) in venueNos:
#         #         #key = matches[element][i]
#         #         matches[element][i] = key
#         # if element == 'toss_decision':
#         #     if (matches[element][i]) in tossNos:
#         #         #key = matches[element][i]
#         #         matches[element][i] = key
#
# matches.to_csv('ipl/matches_codes.csv',index=False)