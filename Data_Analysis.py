import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt

matchesdata = pd.read_csv('ipl/matches_new.csv')
deliveriesdata = pd.read_csv('ipl/deliveries_new.csv')

print("Matches File Details")
print(matchesdata.describe())
print("Deliveries File Details")
print(deliveriesdata.describe())
matchesdata.describe().to_csv('ipl/describematches.csv')
deliveriesdata.describe().to_csv('ipl/describedeliveries.csv')
sea.countplot(x='season' , data = matchesdata)
plt.ylabel("No. of Matches")
plt.show()

#data_2018 = pd.read_csv('ipl/matches_codes1.csv')




