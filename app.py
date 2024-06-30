from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import plotly.graph_objects as go
import pickle
import os
from datetime import datetime
from flask import session

app = Flask(__name__)
app.secret_key = '362655941d39c6745054b3e55d5dac8a'

# Constants for Visa benefits
VISA_DISCOUNT_AMOUNT = 0.33
VISA_CASHBACK_AMOUNT = 0.26
VISA_REWARD_POINTS = 100
VISA_EXCLUSIVE_OFFER = "10% off on next billing cycle"


# Constants for environmental impact calculations
CARBON_EMISSION_PER_KWH = 0.5  # in kg CO2
CARBON_EMISSION_REDUCTION_FACTOR = 0.1

# Load or initialize historical data storage
historical_data_path = 'historical_energy_consumption.csv'
if os.path.exists(historical_data_path):
    historical_data_df = pd.read_csv(historical_data_path)
else:
    historical_data_df = pd.DataFrame(columns=['User_ID', 'Country', 'Month', 'Year', 'Date', 'Appliances', 'Predicted_Energy_kWh'])

# Load the datasets
energy = pd.read_csv('Household_energy_data.csv')
govt_schemes = pd.read_csv('govt schemes.csv')
energy_providers = pd.read_csv('energy_providers.csv')

# Sample and preprocess energy data
energy = energy.sample(10000, random_state=15)
energy['Year'] = pd.to_datetime(energy['Year'], format='%Y').dt.strftime('%Y')
energy['Month'] = pd.to_datetime(energy['Month'], format='%b').dt.strftime('%b')
energy['Household_ID'] = energy['Household_ID'].astype(str)
energy_con = energy.drop(columns=['Total_Consumption_kWh'])

# Create dummy variables or preprocess data further as needed
data = pd.get_dummies(energy_con, columns=['Country', 'Appliance', 'Month', 'Year'])
data = data.drop(columns=['Household_ID'])

# Define features and target
Y = data['Energy_Consumption_kWh']
X = data.drop(columns=['Energy_Consumption_kWh'])

# Load the model, scaler, and column names
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
with open('columns.pkl', 'rb') as columns_file:
    columns = pickle.load(columns_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global historical_data_df
    
    user_id = request.form['user_id']
    
    # Country validation
    country = request.form['country'].strip().capitalize()
    if country not in energy['Country'].unique():
        error_message = "Invalid country. Please enter a valid country from the list."
        return error_message

    # Check if user is registered with different country data
    user_country_history = historical_data_df[(historical_data_df['User_ID'] == user_id) & (historical_data_df['Country'] != country)]
    if not user_country_history.empty:
        error_message = f"User already registered with different country data: {user_country_history['Country'].unique()}"
        return error_message
    
    # Month validation
    month = request.form['month'].strip().capitalize()
    valid_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    if month not in valid_months:
        error_message = "Invalid month. Please enter a valid month abbreviation (e.g., Jan)."
        return error_message

    # Year validation
    year = request.form['year'].strip()
    if not (year.isdigit() and 2010 <= int(year) <= 2100):
        error_message = "Invalid year. Please enter a valid year (between 2010 and 2100)."
        return error_message

    # Date validation
    date = request.form['date'].strip()
    try:
        datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        error_message = "Invalid date format. Please enter a valid date (YYYY-MM-DD)."
        return error_message

    # Number of appliances validation
    try:
        num_appliances = int(request.form['num_appliances'])
        if num_appliances <= 0:
            raise ValueError("Number of appliances must be greater than zero.")
    except ValueError:
        error_message = "Invalid input. Please enter a valid number of appliances."
        return error_message

    # Appliance validation
    appliances = []
    for i in range(num_appliances):
        appliance = request.form[f'appliance_{i+1}'].strip().capitalize()
        if appliance in energy['Appliance'].unique():
            appliances.append(appliance)
        else:
            error_message = "Invalid appliance. Please enter a valid appliance from the list."
            return error_message

    # Energy cost validation
    try:
        current_energy_cost_per_kwh = float(request.form['current_energy_cost_per_kwh'])
        if current_energy_cost_per_kwh <= 0:
            raise ValueError("Energy cost must be greater than zero.")
    except ValueError:
        error_message = "Invalid input. Please enter a valid energy cost per kWh."
        return error_message

    total_prediction = 0
    max_energy_consumption = 0
    max_energy_appliance = ""

    for appliance in appliances:
        input_data = pd.DataFrame({
            'Country': [country],
            'Month': [month],
            'Year': [year],
            'Appliance': [appliance]
        })
        input_data = pd.get_dummies(input_data, columns=['Country', 'Appliance', 'Month', 'Year'])
        input_data = input_data.reindex(columns=columns, fill_value=0)
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        total_prediction += prediction[0]

        if prediction[0] > max_energy_consumption:
            max_energy_consumption = prediction[0]
            max_energy_appliance = appliance

    # Append to historical data
    new_data = pd.DataFrame({
        'User_ID': [user_id],
        'Country': [country],
        'Month': [month],
        'Year': [year],
        'Date': [date],
        'Appliances': [', '.join(appliances)],
        'Predicted_Energy_kWh': [total_prediction]
    })
    historical_data_df = pd.concat([historical_data_df, new_data], ignore_index=True)
    historical_data_df.to_csv(historical_data_path, index=False)

    # Display government schemes for the selected country
    applicable_schemes = govt_schemes[govt_schemes['Country'] == country]['Scheme/Subsidy'].tolist()

    return render_template('result.html', 
                           user_id=user_id, 
                           country=country, 
                           month=month, 
                           year=year, 
                           date=date, 
                           appliances=appliances, 
                           total_prediction=total_prediction, 
                           max_energy_consumption=max_energy_consumption, 
                           max_energy_appliance=max_energy_appliance, 
                           current_energy_cost_per_kwh=current_energy_cost_per_kwh, 
                           applicable_schemes=applicable_schemes)

@app.route('/history', methods=['GET', 'POST'])
def history():
    if request.method == 'POST':
        user_id = request.form['user_id']
    else:
        user_id = request.args.get('user_id')

    user_history = historical_data_df[historical_data_df['User_ID'] == user_id]

    if user_history.empty:
        error_message = "No historical data found for the user."
        return render_template('index.html', error_message=error_message)

    user_history['Date'] = pd.to_datetime(user_history['Date'], format='%Y-%m-%d')
    user_history = user_history.dropna(subset=['Date'])

    try:
        fig = go.Figure()

        # Add historical data as bar plot
        fig.add_trace(go.Bar(
            x=user_history['Date'],
            y=user_history['Predicted_Energy_kWh'],
            marker_color='blue',
            name='Historical Data',
            hovertext=user_history.apply(lambda row: f"{row['Date'].strftime('%Y-%m-%d')}<br>{row['Predicted_Energy_kWh']} kWh", axis=1)
        ))

        # Add current prediction as a vertical line at the correct date
        current_date = user_history['Date'].max()
        current_prediction = user_history[user_history['Date'] == current_date]['Predicted_Energy_kWh'].values[0]
        fig.add_trace(go.Scatter(
            x=[current_date, current_date],
            y=[0, current_prediction],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Current Prediction',
            hovertext='Current Prediction'
        ))

        # Update layout
        fig.update_layout(
            title=f'Historical vs Current Energy Consumption for User ID: {user_id}',
            xaxis_title='Date',
            yaxis_title='Energy Consumption (kWh)',
            xaxis=dict(tickformat='%Y-%m-%d'),
            xaxis_tickangle=-45,
            showlegend=True,
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255, 255, 255, 0.6)'),
            margin=dict(l=40, r=40, t=80, b=80),
            hovermode='x'
        )

        graph_html = fig.to_html(full_html=False)

    except Exception as e:
        error_message = f"An error occurred while generating the historical data visualization: {str(e)}"
        return render_template('index.html', error_message=error_message)

    return render_template('historical_data.html', graph_html=graph_html)

@app.route('/alternate_energy', methods=['POST'])
def alternate_energy():
    user_id = request.form['user_id']
    selected_option = request.form['alternate_energy']
    current_energy_cost_per_kwh = request.form['current_energy_cost_per_kwh']

    if selected_option.lower() == 'yes':
        # Get appliances used by the user
        user_appliances = historical_data_df[historical_data_df['User_ID'] == user_id]['Appliances'].iloc[-1].split(', ')

        # Filter energy providers based on user appliances
        valid_providers = []
        for index, row in energy_providers.iterrows():
            provider_appliances = row['Appliance'].split(', ')
            if any(appliance in user_appliances for appliance in provider_appliances):
                valid_providers.append(row)

        return render_template('energy_providers.html', 
                               user_id=user_id, 
                               providers=valid_providers, 
                               current_energy_cost_per_kwh=current_energy_cost_per_kwh)
    else:
        message = "No alternate energy sources requested."
        return render_template('index.html', message=message)

@app.route('/select_provider', methods=['POST'])
def select_provider():
    user_id = request.form['user_id']
    provider_name = request.form['provider']
    current_energy_cost_per_kwh = float(request.form['current_energy_cost_per_kwh'])

    # Find the selected provider details
    selected_provider = energy_providers[energy_providers['Provider'] == provider_name].iloc[0]

    # Calculate total_prediction (assuming it's calculated earlier in your application flow)
    total_prediction = historical_data_df[historical_data_df['User_ID'] == user_id]['Predicted_Energy_kWh'].iloc[-1]

    # Access 'Cost per kWh' correctly
    try:
        cost_per_kwh = selected_provider['Cost_per_kWh']
        alternate_energy_cost = total_prediction * cost_per_kwh

        # Store alternate_energy_cost in session
        session['alternate_energy_cost'] = alternate_energy_cost

        # Other calculations as needed
        current_energy_cost = total_prediction * current_energy_cost_per_kwh
        cost_savings = current_energy_cost - alternate_energy_cost

        current_carbon_emissions = total_prediction * CARBON_EMISSION_PER_KWH
        reduced_carbon_emissions = current_carbon_emissions * CARBON_EMISSION_REDUCTION_FACTOR

        return render_template('selected_provider.html', 
                               user_id=user_id, 
                               provider=selected_provider, 
                               current_energy_cost=current_energy_cost, 
                               alternate_energy_cost=alternate_energy_cost, 
                               cost_savings=cost_savings, 
                               current_carbon_emissions=current_carbon_emissions, 
                               reduced_carbon_emissions=reduced_carbon_emissions)
    except KeyError:
        error_message = f"Selected provider '{provider_name}' does not have 'Cost per kWh' information."
        return render_template('index.html', error_message=error_message)
    
@app.route('/apply_visa_benefits', methods=['POST'])
def apply_visa_benefits():
    # Access alternate_energy_cost from session
    alternate_energy_cost = session.get('alternate_energy_cost')

    if alternate_energy_cost is None:
        error_message = "Alternate energy cost not found in session. Please select an energy provider first."
        return render_template('index.html', error_message=error_message)

    # Handle applying Visa benefits based on user selection
    visa_applied = request.form.get('use_visa', 'no')  # Check if user wants to use Visa

    if visa_applied.lower() == 'yes':
        # Apply Visa benefits
        discount_amount = alternate_energy_cost * VISA_DISCOUNT_AMOUNT
        cashback_amount = alternate_energy_cost * VISA_CASHBACK_AMOUNT
        final_energy_cost_with_visa = alternate_energy_cost - discount_amount - cashback_amount

        return render_template('visa_benefits.html', 
                               discount_amount=discount_amount, 
                               cashback_amount=cashback_amount, 
                               reward_points=VISA_REWARD_POINTS, 
                               exclusive_offer=VISA_EXCLUSIVE_OFFER, 
                               final_energy_cost_with_visa=final_energy_cost_with_visa)
    else:
        # Proceed without applying Visa benefits
        return render_template('thank_you.html')



if __name__ == '__main__':
    app.run(debug=True)
