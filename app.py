from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model and datasets
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
energy = pd.read_csv("Household_energy_data.csv")
schemes = pd.read_csv("govt schemes.csv")
alt_energy = pd.read_csv("alternate_energy_sources.csv")

# Preprocess the data to handle case and whitespace issues
energy['Country'] = energy['Country'].str.strip().str.capitalize()
energy['Appliance'] = energy['Appliance'].str.strip().str.capitalize()

# Dummy variables for encoding
countries = list(energy['Country'].unique())
appliances = list(energy['Appliance'].unique())
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
years = [str(year) for year in range(2010, 2024)]

# Function to predict energy consumption
def predict_energy_consumption(country, month, year, appliances):
    total_prediction = 0
    for appliance in appliances:
        input_data = pd.DataFrame({
            'Country': [country],
            'Month': [month],
            'Year': [year],
            'Appliance': [appliance]
        })
        input_data = pd.get_dummies(input_data, columns=['Country', 'Appliance', 'Month', 'Year'])
        input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        total_prediction += prediction[0]
    return total_prediction

@app.route('/')
def home():
    return render_template('index.html', countries=countries, months=months, years=years, appliances=appliances)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        country = data.get('country', '').strip().capitalize()
        month = data.get('month', '').strip().capitalize()
        year = data.get('year', '').strip()
        num_appliances = int(data.get('num_appliances', 0))
        appliances = [data.get(f'appliance{i+1}', '').strip().capitalize() for i in range(num_appliances)]

        # Validate inputs
        valid_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        if not country or country not in countries:
            return jsonify({'error': 'Invalid country. Please enter a valid country from the list.'}), 400

        if not month or month not in valid_months:
            return jsonify({'error': 'Invalid month. Please enter a valid month abbreviation (e.g., Jan).'}), 400

        if not year.isdigit() or not (2010 <= int(year) <= 2023):
            return jsonify({'error': 'Invalid year. Please enter a valid year (between 2010 and 2023).'}), 400

        if num_appliances <= 0:
            return jsonify({'error': 'Number of appliances must be greater than zero.'}), 400

        for appliance in appliances:
            if appliance not in appliances:
                return jsonify({'error': f'Invalid appliance: {appliance}. Please enter a valid appliance from the list.'}), 400

        # Predict energy consumption
        prediction = predict_energy_consumption(country, month, year, appliances)
        prediction_rounded = round(prediction, 2)

        # Suggest alternate energy sources
        filtered_alt = alt_energy[(alt_energy['Country'] == country) &
                                  (alt_energy['Min Consumption (kWh)'] <= prediction) &
                                  (alt_energy['Max Consumption (kWh)'] >= prediction)].sort_values('Avg Installation Cost ($)')

        relevant_schemes = schemes[schemes['Country'] == country][['Scheme/Subsidy', 'Specific Source']]

        return jsonify({
            'prediction': prediction_rounded,
            'alternate_sources': filtered_alt.to_dict(orient='records'),
            'government_schemes': relevant_schemes.to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
