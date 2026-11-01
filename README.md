# Developing a Neural Network Classification Model

---

### Name: Sanjeev A

### Register Number: 212224230246

---

## AIM

To develop a neural network classification model for the given dataset.

---

## THEORY

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for different segments of customers. This strategy has worked exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of new customers.

---

## Neural Network Model

<img width="977" height="858" alt="image" src="https://github.com/user-attachments/assets/61c686c7-0867-4176-aee5-50172758eb16" />

---

## DESIGN STEPS

### STEP 1: 
Load and Preprocess the Dataset:

Import the required libraries, load the dataset, handle missing values, encode categorical variables, and normalize the features.

### STEP 2: 

Split the Dataset:

Divide the dataset into training and testing sets.

### STEP 3: 

Build the Neural Network:

Create a Sequential model with an input layer, hidden layers using ReLU activation, and an output layer with Softmax activation for the four customer segments.

### STEP 4: 

Train the Model:

Compile the model using the Adam optimizer and categorical cross-entropy loss function, then train it using the training data.

### STEP 5: 

Evaluate and Predict:

Evaluate the model on the test dataset, measure its accuracy, and use it to predict the customer segment (A, B, C, or D) for new customers.



---



## PROGRAM

```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from torch.utils.data import TensorDataset,DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv(r"C:\Users\sarav\Downloads\customers (1).csv")
df
df=df.drop(columns=["ID"])
df
df.columns
df.fillna({"Work_Experience":0,"Family_Size":df["Family_Size"].median()},inplace=True)
df
cat_columns=['Gender','Ever_Married', 'Graduated', 'Profession',
    'Spending_Score', 'Var_1']
for col in cat_columns:
    df[col]=LabelEncoder().fit_transform(df[col])
lbe=LabelEncoder()
df["Segmentation"]=lbe.fit_transform(df["Segmentation"])
df
x=df.drop(columns="Segmentation")
y=df["Segmentation"].values
xt,xst,yt,yst=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
xt=scaler.fit_transform(xt)
xst=scaler.transform(xst)
xt=torch.FloatTensor(xt)
xst=torch.FloatTensor(xst)
yt=torch.FloatTensor(yt)
yst=torch.FloatTensor(yst)
tr=TensorDataset(xt,yt)
tst=TensorDataset(xst,yst)
trl=DataLoader(tr,batch_size=16,shuffle=True)
tstl=DataLoader(tst,batch_size=16)
class classifier1(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.l1=nn.Linear(input_size,64)
        self.l2=nn.Linear(64,32)
        self.l3=nn.Linear(32,16)
        self.l4=nn.Linear(16,4)
        self.dropout=nn.Dropout(0.2)

    def forward(self,x):
        x=F.relu(self.l1(x))
        x=self.dropout(x)
        x=F.relu(self.l2(x))
        x=self.dropout(x)
        x=F.relu(self.l3(x))
        x=self.l4(x)
        return x
model=classifier1(input_size=xt.shape[1])

criterion=nn.CrossEntropyLoss()

op=optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5
)

scheduler=optim.lr_scheduler.ReduceLROnPlateau(
    op,
    mode='max',
    factor=0.5,
    patience=10
)
epochs=200

for i in range(epochs):
    model.train()

    for a,b in trl:
        op.zero_grad()

        pred=model(a)
        loss=criterion(pred,b.long())

        loss.backward()
        op.step()

    model.eval()

    with torch.no_grad():
        output=model(xst)
        _,predicted=torch.max(output,1)
        val_accuracy=accuracy_score(
            yst.numpy(),
            predicted.numpy()
        )

    scheduler.step(val_accuracy)

    if i%10==0:
        print(
            f"Epoch:{i}/{epochs} "
            f"Loss:{loss.item():.4f} "
            f"Accuracy:{val_accuracy:.4f}"
        )
mpre=[]
act=[]

model.eval()

with torch.no_grad():
    output=model(xst)
    _,predicted=torch.max(output,1)

    pre.extend(predicted.numpy())
    act.extend(yst.numpy())

accuracy=accuracy_score(act,pre)
conf_matrix=confusion_matrix(act,pre)
cl_report=classification_report(
    act,
    pre,
    target_names=['A','B','C','D']
)

print("Accuracy:",accuracy)
print("confusion_matrix:\n",conf_matrix)
print("classification_report:\n",cl_report)
accuracy=accuracy_score(act,pre)
conf_matrix=confusion_matrix(act,pre)

print("Accuracy:",accuracy)
print("confusion_matrix:\n",conf_matrix)

print("classification_report:")
print(
    classification_report(
        act,
        pre,
        target_names=['A','B','C','D'],
        zero_division=0
    )
)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

model.eval()

with torch.no_grad():
    output = model(xst)
    _, predicted = torch.max(output, 1)

act = yst.numpy()
pre = predicted.numpy()

accuracy = accuracy_score(act, pre)
conf_matrix = confusion_matrix(act, pre)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(
    act,
    pre,
    target_names=['A', 'B', 'C', 'D']
))
import seaborn as sns
import matplotlib.pyplot as plt

labels = ['A', 'B', 'C', 'D']

plt.figure(figsize=(7, 5))

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Customer Segmentation - Confusion Matrix")
plt.tight_layout()
plt.show()
```

---

### OUTPUT

## Confusion Matrix
<img width="878" height="662" alt="Screenshot 2026-11-01 222031" src="https://github.com/user-attachments/assets/d8555a1b-e444-4cc6-ade0-2d93d0b1a7fc" />

## Classification Report
<img width="610" height="479" alt="Screenshot 2026-11-01 222011" src="https://github.com/user-attachments/assets/55280ef9-7d6a-4d08-b062-35b86244cd46" />


---

## RESULT
This program has been executed successfully.
