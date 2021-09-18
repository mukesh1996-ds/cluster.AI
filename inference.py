import pickle

model = pickle.load(open('insurance.pkl', 'rb'))

def predict(data):
    prediction = model.predict(data)
    return prediction

"""Output"""

cost = predict([[19,0,27.9,0,1,0 ]])
print(cost)