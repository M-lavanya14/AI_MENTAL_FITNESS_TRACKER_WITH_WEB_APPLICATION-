from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the machine learning model and preprocess the data
df1 = pd.read_csv("mental-and-substance-use-as-share-of-disease.csv")
df2 = pd.read_csv("prevalence-by-mental-and-substance-use-disorder.csv")
common_columns = list(set(df1.columns) & set(df2.columns))
data = pd.merge(df1, df2, on=common_columns)
data.drop(['Code'], axis=1, inplace=True)
data.columns = ['Country', 'Year', 'Schizophrenia', 'Bipolar_disorder', 'Eating_disorder', 'Anxiety', 'Drug_usage', 'Depression', 'Alcohol', 'Mental_fitness']
le = LabelEncoder()
for col in ['Country']:
    data[col] = le.fit_transform(data[col])

X = data.drop('Mental_fitness', axis=1)
y = data['Mental_fitness']

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf = RandomForestRegressor(random_state=42)
rf.fit(xtrain, ytrain)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for handling the form submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        country = request.form['country']
        year = int(request.form['year'])
        schi = float(request.form['schi'])
        bipo_dis = float(request.form['bipo_dis'])
        eat_dis = float(request.form['eat_dis'])
        anx = float(request.form['anx'])
        drug_use = float(request.form['drug_use'])
        depr = float(request.form['depr'])
        alch = float(request.form['alch'])

        # Encode country using LabelEncoder
        country_encoded = le.transform([country])[0]

        # Make prediction
        input_values = [[country_encoded, year, schi, bipo_dis, eat_dis, anx, drug_use, depr, alch]]
        prediction = rf.predict(input_values)

        # Display input and predicted values for debugging
        print(f"Input Values: {input_values}")
        print(f"Predicted Mental Fitness: {prediction[0] * 10}")

        # Render the result template with the predicted value
        return render_template('result.html', prediction=prediction[0] * 10)

if __name__ == '__main__':
    app.run(debug=True)
