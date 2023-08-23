from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

file_dataFrame=pd.read_csv('archive\loan-train.csv').dropna()

income_primary=file_dataFrame['ApplicantIncome']
load_amount=file_dataFrame['LoanAmount']
# plt.scatter(income_primary, load_amount)
# plt.title('Loan Provision Basic Data Plot')
# plt.xlabel('Income')
# plt.ylabel('Loan')
# plt.show()

data = list(zip(income_primary, load_amount))


for i in range(1,11):
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(data)
    

plt.title('Loan Provision')
plt.xlabel('Income')
plt.ylabel('Loan')
plt.scatter(income_primary, load_amount, c=kmeans.labels_)
plt.show()

