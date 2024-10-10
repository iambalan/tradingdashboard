import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import streamlit as st
import matplotlib.pyplot as plt
import re

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_excel('https://drive.google.com/uc?id=13lOtGbM4-NgOXppI90yjuZVGT_ijhUSG')

def mts_conversion(df):
    conversion_dict = {
    'KGS': 1/1000,  # Kilograms to metric tons
    'KGA': 1/1000,  # Assuming KGA is the same as KGS
    'LBS': 1/2204.62,  # Pounds to metric tons
    'MTS': 1,  # Already in metric tons
    'NOS': np.nan,  # Not convertible to metric tons
    'MTR': np.nan,  # Not convertible to metric tons
    'CBM': np.nan,  # Not convertible to metric tons
    'KLR': np.nan   # Not convertible to metric tons
    }

    # Remove trailing spaces in UQC
    df['UQC'] = df['UQC'].str.strip()
    
    # Convert quantities based on UQC
    df['QUANTITY_MTS'] = df.apply(lambda row: row['QUANTITY'] * conversion_dict.get(row['UQC'], np.nan), axis=1)
    
    # Filter out non-convertible rows if needed
    df = df[df['QUANTITY_MTS'].notna()]
    return df


def port_country_agg(df):
    # Define a dictionary to map alternative names to a standard version
    country_corrections = {
        "Côte D'Ivoire": "COTE D'IVOIRE",
        "Cote D Ivoire ": "COTE D'IVOIRE",
        "CÔTE D'IVOIRE": "COTE D'IVOIRE",
        "GHANA ": "GHANA",
        "Ghana ": "GHANA",
        "Tanzania ": "TANZANIA",
        "Mozambique ": "MOZAMBIQUE",
        "Madagascar ": "MADAGASCAR",
        "Burkina Faso ": "BURKINA FASO",
        "Vietnam, Democratic Rep. Of ": "VIETNAM",
        "Nigeria ": "NIGERIA",
        "Guinea-Bissau": "GUINEA-BISSAU",
        "Gambia ": "GAMBIA",
        "Indonesia ": "INDONESIA",
        "Thailand ": "THAILAND",
        "Myanmar ": "MYANMAR"
    }
    
    # Strip spaces and convert to upper case
    df['FOREIGN COUNTRY'] = df['FOREIGN COUNTRY'].str.strip().str.upper()
    
    # Apply manual corrections using the dictionary
    df['FOREIGN COUNTRY'] = df['FOREIGN COUNTRY'].replace(country_corrections)

    # Extend the dictionary to handle more complex port names
    port_corrections = {
        'TUTICORIN SEA': 'TUTICORIN',
        'VIZAG SEA': 'VIZAG',
        'KOLKATA SEA': 'KOLKATA',
        'COCHIN SEA': 'COCHIN',
        'TUTICORIN ICD': 'TUTICORIN',
        'MUNDRA SEA': 'MUNDRA',
        'MANGALORE SEA': 'MANGALORE',
        'CHENNAI SEA': 'CHENNAI',
        'KANDLA-SEZ': 'KANDLA',
        'MUNDRA-SEZ': 'MUNDRA',
        'COCHIN AIR': 'COCHIN',
        'CHENNAI AIR': 'CHENNAI',
        'DELHI AIR CARGO': 'DELHI',
        'BOMBAY AIR CARGO': 'MUMBAI',
        'MUNDRA': 'MUNDRA',
        'KANDLA': 'KANDLA',
        'JNPT': 'JAWAHARLAL NEHRU PORT',
        'TUGHLAKABAD': 'TUGHLAKABAD',
        'LUDHIANA GRFL SAHNEWAL ICD': 'LUDHIANA',
        'LUDHIANA ICD DHANDARI KALAN': 'LUDHIANA',
        'KANAKPURA ICD': 'KANAKPURA',
        'BHUSAWAL ICD': 'BHUSAWAL',
        'AHMEDABAD (SABARMATI) ICD': 'AHMEDABAD',
        'KANPUR - JRY (ICD )': 'KANPUR',
        'PITHAMPUR': 'PITHAMPUR',
        'BARHI ICD': 'BARHI',
        'KILARAIPUR ICD': 'KILARAIPUR',
        'BANGALORE ICD': 'BANGALORE',
        'DADRI-CGML': 'DADRI',
        'DADRI - STTPL (CFS)': 'DADRI',
        'PANCHI GUJARAN, SONEPAT ICD': 'SONEPAT',
        'BHAGAT KI KOTHI - JODHPUR ICD': 'JODHPUR',
        'JAIPUR ICD': 'JAIPUR',
        'JATTIPUR ICD': 'JATTIPUR',
        'KATTUPALLI VILLAGE,PONNERI TALUK,TIRUVALLUR': 'KATTUPALLI',
        'KRISHNAPATNAM': 'KRISHNAPATNAM',
        'GOA SEA': 'GOA',
        'CHAWAPAYAL ICD/SAMRALA': 'SAMRALA',
        'CONCOR ICD MIHAN': 'MIHAN',
        'SRIMANTAPUR LCS': 'SRIMANTAPUR',
        'LUDHIANA ICD' : 'LUDIHANA'
        # Add more corrections as necessary
    }

    # Strip spaces and convert to upper case for uniformity
    df['DOMESTIC PORT'] = df['DOMESTIC PORT'].str.strip().str.upper()
    
    # Apply manual corrections using the updated dictionary
    df['DOMESTIC PORT'] = df['DOMESTIC PORT'].replace(port_corrections)

    # Define a dictionary to map alternative foreign port names to a standard version
    foreign_port_corrections = {
        'TAMATAVE (TOAMASINA)': 'TOAMASINA',
        'TINCAN/LAGOS': 'TINCAN LAGOS',
        'TINCAN LAGOS': 'TINCAN LAGOS',
        'MAJUNGA (MAHAJANGA)': 'MAHAJANGA',
        'DAR ES SALAM (DARESS)': 'DAR ES SALAAM',
        'HO CHI MINH CITY': 'HO CHI MINH',
        'HO CHI MINH C': 'HO CHI MINH',
        'HO CHI MINH, VICT': 'HO CHI MINH',
        'ANTSIRANANA': 'ANTISIRANANA',
        'BOMHRA': 'BHOMRA',
        'LAEM CHABANG': 'LAEM CHABANG',
        'LAT KRABANG': 'LAT KRABANG',
        'FELIXSTOWE': 'FELIXSTOWE',
        'DAR ES SALAAM': 'DAR ES SALAAM',
        'COMILLA': 'COMILLA',
        'ZIGUINCHOR': 'ZIGUINCHOR',
        'SURABAYA': 'SURABAYA',
        'TEMA': 'TEMA',
        # Add more corrections based on inspection
    }
    
    # Strip spaces, remove any extra whitespace, and convert to uppercase for uniformity
    df['FOREIGN PORT'] = df['FOREIGN PORT'].str.strip().str.upper()
    
    # Apply the corrections using the dictionary
    df['FOREIGN PORT'] = df['FOREIGN PORT'].replace(foreign_port_corrections)

    return df

def top_ports_by_total_usd():

    dftemp = df.copy()
    # Step 7: Filter data for selected out turns if not 'All'
    if 'All' not in selected_out_turns:
        dftemp = dftemp[dftemp['Out Turn (Lbs)'].isin(selected_out_turns)]
    else:
        dftemp = dftemp.copy()

    # Step 8: Filter data for selected nut counts if not 'All'
    if 'All' not in selected_nut_counts:
        dftemp = dftemp[dftemp['Nut Count'].isin(selected_nut_counts)]
    
        
    port_weights = dftemp.groupby('FOREIGN PORT',as_index=False)['TOTAL VALUE USD'].sum()
    # Treemap visualization
    fig = px.treemap(port_weights, path=['FOREIGN PORT'], values='TOTAL VALUE USD',
                     title='Total USD Distribution by Port of Loading',
                     labels={'TOTAL VALUE USD': 'TOTAL USD'})
    st.plotly_chart(fig)


def top_ports_by_usd_per_mt():

    dftemp = df.copy()
    # Step 7: Filter data for selected out turns if not 'All'
    if 'All' not in selected_out_turns:
        dftemp = dftemp[dftemp['Out Turn (Lbs)'].isin(selected_out_turns)]
    else:
        dftemp = dftemp.copy()

    # Step 8: Filter data for selected nut counts if not 'All'
    if 'All' not in selected_nut_counts:
        dftemp = dftemp[dftemp['Nut Count'].isin(selected_nut_counts)]

    port_weights = dftemp.groupby('FOREIGN PORT',as_index=False)['DOMESTIC USDMT'].mean()
    # Treemap visualization
    fig = px.treemap(port_weights, path=['FOREIGN PORT'], values='DOMESTIC USDMT',
                     title='Domestic per unit USD/MT',
                     labels={'DOMESTIC USDMT': 'Domestic USD/MT'})
    st.plotly_chart(fig)

def flow_by_totalusd():

    dftemp = df.copy()
    # Step 7: Filter data for selected out turns if not 'All'
    if 'All' not in selected_out_turns:
        dftemp = dftemp[dftemp['Out Turn (Lbs)'].isin(selected_out_turns)]
    else:
        dftemp = dftemp.copy()

    # Step 8: Filter data for selected nut counts if not 'All'
    if 'All' not in selected_nut_counts:
        dftemp = dftemp[dftemp['Nut Count'].isin(selected_nut_counts)]
    
    # Sample data preparation - group by the flow between Origin Country, Port of Loading, and Port of Discharge
    # Adjust as per your DataFrame names and content
    sankey_data = dftemp.groupby(['FOREIGN COUNTRY', 'FOREIGN PORT', 'DOMESTIC PORT'],as_index=False)['TOTAL VALUE USD'].mean().reset_index(drop=True)
    
    # Filter to show only top flows to reduce clutter
    # Display top 10 by weight; adjust this number as needed to simplify the diagram
    top_flows = sankey_data.sort_values(by='TOTAL VALUE USD', ascending=False).head(50)
    
    # Create node labels list
    nodes = pd.concat([top_flows['FOREIGN COUNTRY'], top_flows['FOREIGN PORT'], top_flows['DOMESTIC PORT']]).unique().tolist()
    
    # Create a mapping of labels to indices
    label_dict = {label: i for i, label in enumerate(nodes)}
    
    # Define sources, targets, and values for the Sankey diagram
    sources = top_flows['FOREIGN COUNTRY'].map(label_dict).tolist() + top_flows['FOREIGN PORT'].map(label_dict).tolist()
    targets = top_flows['FOREIGN PORT'].map(label_dict).tolist() + top_flows['DOMESTIC PORT'].map(label_dict).tolist()
    values = top_flows['TOTAL VALUE USD'].tolist() * 2  # Duplicate for the two-step flow
    
    # Create the Sankey diagram
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color="blue"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(150, 150, 150, 0.4)"  # Set link transparency and color
        )
    ))
    
    fig.update_layout(
        title_text="Flow by Total USD",
        font_size=12,
        height=800,  # Adjust height to give more room to labels
        width=1200   # Adjust width to improve readability
    )

    st.plotly_chart(fig)

def flow_by_usd_per_mt():

    dftemp = df.copy()
    # Step 7: Filter data for selected out turns if not 'All'
    if 'All' not in selected_out_turns:
        dftemp = dftemp[dftemp['Out Turn (Lbs)'].isin(selected_out_turns)]
    else:
        dftemp = dftemp.copy()

    # Step 8: Filter data for selected nut counts if not 'All'
    if 'All' not in selected_nut_counts:
        dftemp = dftemp[dftemp['Nut Count'].isin(selected_nut_counts)]
    
    # Sample data preparation - group by the flow between Origin Country, Port of Loading, and Port of Discharge
    # Adjust as per your DataFrame names and content
    sankey_data = dftemp.groupby(['FOREIGN COUNTRY', 'FOREIGN PORT', 'DOMESTIC PORT'],as_index=False)['DOMESTIC USDMT'].mean().reset_index(drop=True)
    
    # Filter to show only top flows to reduce clutter
    # Display top 10 by weight; adjust this number as needed to simplify the diagram
    top_flows = sankey_data.sort_values(by='DOMESTIC USDMT', ascending=False).head(50)
    
    # Create node labels list
    nodes = pd.concat([top_flows['FOREIGN COUNTRY'], top_flows['FOREIGN PORT'], top_flows['DOMESTIC PORT']]).unique().tolist()
    
    # Create a mapping of labels to indices
    label_dict = {label: i for i, label in enumerate(nodes)}
    
    # Define sources, targets, and values for the Sankey diagram
    sources = top_flows['FOREIGN COUNTRY'].map(label_dict).tolist() + top_flows['FOREIGN PORT'].map(label_dict).tolist()
    targets = top_flows['FOREIGN PORT'].map(label_dict).tolist() + top_flows['DOMESTIC PORT'].map(label_dict).tolist()
    values = top_flows['DOMESTIC USDMT'].tolist() * 2  # Duplicate for the two-step flow
    
    # Create the Sankey diagram
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color="blue"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(150, 150, 150, 0.4)"  # Set link transparency and color
        )
    ))
    
    fig.update_layout(
        title_text="Flow by USD PER MT",
        font_size=12,
        height=800,  # Adjust height to give more room to labels
        width=1200   # Adjust width to improve readability
    )

    st.plotly_chart(fig)

def usd_per_mt_monthly():

    dftemp = df.copy()
    # Step 7: Filter data for selected out turns if not 'All'
    if 'All' not in selected_out_turns:
        dftemp = dftemp[dftemp['Out Turn (Lbs)'].isin(selected_out_turns)]
    else:
        dftemp = dftemp.copy()

    # Step 8: Filter data for selected nut counts if not 'All'
    if 'All' not in selected_nut_counts:
        dftemp = dftemp[dftemp['Nut Count'].isin(selected_nut_counts)]
    
    # Step 1: Convert 'Date' to datetime if not already
    dftemp['BILL DATE'] = pd.to_datetime(dftemp['BILL DATE'])
    
    # Step 2: Group by month and country, summing the weights
    monthly_weights = dftemp.groupby([dftemp['BILL DATE'].dt.to_period('Y'), 'FOREIGN COUNTRY'])['DOMESTIC USDMT'].mean().reset_index()
    
    # Convert 'Date' back to timestamp for Plotly
    monthly_weights['BILL DATE'] = monthly_weights['BILL DATE'].dt.to_timestamp()
    
    # Step 3: Create Plotly Visualization
    fig = px.line(
        monthly_weights,
        x='BILL DATE',
        y='DOMESTIC USDMT',
        color='FOREIGN COUNTRY',
        title='Monthly DOMESTIC USD/MT by Country',
        labels={'DOMESTIC USDMT': 'DOMESTIC USD/MT', 'BILL DATE': 'Month'},
        markers=True
    )

    st.plotly_chart(fig)

def total_usd_monthly():

    dftemp = df.copy()
    # Step 7: Filter data for selected out turns if not 'All'
    if 'All' not in selected_out_turns:
        dftemp = dftemp[dftemp['Out Turn (Lbs)'].isin(selected_out_turns)]
    else:
        dftemp = dftemp.copy()

    # Step 8: Filter data for selected nut counts if not 'All'
    if 'All' not in selected_nut_counts:
        dftemp = dftemp[dftemp['Nut Count'].isin(selected_nut_counts)]
    
    # Step 1: Convert 'Date' to datetime if not already
    dftemp['BILL DATE'] = pd.to_datetime(dftemp['BILL DATE'])
    
    # Step 2: Group by month and country, summing the weights
    monthly_weights = dftemp.groupby([dftemp['BILL DATE'].dt.to_period('M'), 'FOREIGN COUNTRY'])['TOTAL VALUE USD'].sum().reset_index()
    
    # Convert 'Date' back to timestamp for Plotly
    monthly_weights['BILL DATE'] = monthly_weights['BILL DATE'].dt.to_timestamp()
    
    # Step 3: Create Plotly Visualization
    fig = px.line(
        monthly_weights,
        x='BILL DATE',
        y='TOTAL VALUE USD',
        color='FOREIGN COUNTRY',
        title='Monthly USD by Country',
        labels={'TOTAL VALUE USD': 'Total USD', 'BILL DATE': 'Month'},
        markers=True
    )


    st.plotly_chart(fig)

def weekly_moving_avg():

    dftemp = df.copy()
    # Step 7: Filter data for selected out turns if not 'All'
    if 'All' not in selected_out_turns:
        dftemp = dftemp[dftemp['Out Turn (Lbs)'].isin(selected_out_turns)]
    else:
        dftemp = dftemp.copy()

    # Step 8: Filter data for selected nut counts if not 'All'
    if 'All' not in selected_nut_counts:
        dftemp = dftemp[dftemp['Nut Count'].isin(selected_nut_counts)]
    
    # Convert the column with the datetime index if not already done
    dftemp['BILL DATE'] = pd.to_datetime(dftemp['BILL DATE'])  # Replace 'ts' with your timestamp column name
    
    # Set the index to the datetime column for resampling
    dftemp.set_index('BILL DATE', inplace=True)
    
    # Resample only the numeric columns weekly and sum them
    weekly_data = dftemp.select_dtypes(include=[int, float]).resample('W').sum().reset_index()
    
    # If you want to include non-numeric columns (e.g., taking the first value for those columns)
    non_numeric_cols = dftemp.select_dtypes(exclude=[int, float])
    weekly_data_non_numeric = non_numeric_cols.resample('W').first().reset_index()
    
    # Merge numeric and non-numeric data
    weekly_data = pd.merge(weekly_data, weekly_data_non_numeric, on='BILL DATE', how='left')

    weekly_data['Moving_Average'] = weekly_data['TOTAL VALUE USD'].rolling(window=3).mean()
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add Weekly Supply trace
    fig.add_trace(go.Scatter(x=weekly_data['BILL DATE'], y=weekly_data['TOTAL VALUE USD'],
                             mode='lines', name='Weekly Supply', 
                             opacity=0.5))
    
    # Add Moving Average trace
    fig.add_trace(go.Scatter(x=weekly_data['BILL DATE'], y=weekly_data['Moving_Average'],
                             mode='lines', name='3-Week Moving Average', 
                             line=dict(color='orange')))
    
    # Update layout
    fig.update_layout(title='Weekly USD Trend Analysis',
                      xaxis_title='BILL DATE',
                      yaxis_title='USD',
                      legend=dict(x=0, y=1))
    
    st.plotly_chart(fig)

# def forecast():

#     if 'All' not in selected_out_turns:
#         dftemp = df_new[df_new['Out Turn (Lbs)'].isin(selected_out_turns)]
#     else:
#         dftemp = df.copy()

#     # Step 8: Filter data for selected nut counts if not 'All'
#     if 'All' not in selected_nut_counts:
#         dftemp = dftemp[dftemp['Nut Count'].isin(selected_nut_counts)]
    
#     dftemp.reset_index(inplace=True)
#     # Step 1: Calculate the top 10 countries by TOTAL USD
#     top_countries = dftemp.groupby('FOREIGN COUNTRY')['TOTAL VALUE USD'].sum().nlargest(10).index

#     # Step 2: Allow user to select multiple countries with a multiselect
#     selected_country = st.multiselect('Select countries', ['All'] + list(top_countries), default='All')

#     # Step 3: Filter data for the selected countries if not 'All'
#     if 'All' in selected_country:
#         df_country = dftemp.copy()  # Don't filter if 'All' is selected
#     else:
#         df_country = dftemp[dftemp['FOREIGN COUNTRY'].isin(selected_country)]
   
#     # # Step 2: Allow user to select a country from the top 10
#     # selected_country = st.selectbox('Select a country', top_countries)  # Dropdown for country selection

#     # # Step 2: Filter data for the selected country
#     # df_country = dftemp[dftemp['FOREIGN COUNTRY'] == selected_country]
    
#     df_country['DOMESTIC USDMT'].dropna(inplace=True)
    
#     # Step 2: Aggregate or resample data over time (assuming 'Date' column exists)
#     df_country['BILL DATE'] = pd.to_datetime(df_country['BILL DATE'])  # Ensure 'Date' is datetime
#     df_country.set_index('BILL DATE', inplace=True)
    
#     df_country['DOMESTIC USDMT'].fillna(0, inplace=True)
#     # Resample data to monthly totals (you can choose weekly or daily based on your data)
#     df_country_resampled = df_country['DOMESTIC USDMT'].resample('M').mean()
#     df_country_resampled.dropna(inplace=True)
#     # Step 3: Fit ARIMA model
#     # Split into training and testing sets
#     train_data = df_country_resampled[:-15]  # Use all but the last 12 months for training
#     test_data = df_country_resampled[-15:]   # The last 12 months for testing

#     # Fit ARIMA model (p, d, q values can be tuned)
#     model = ARIMA(train_data, order=(7, 1, 9))
#     model_fit = model.fit()

#     # Step 4: Forecast future values
#     forecast = model_fit.forecast(steps=15)  # Forecasting 12 months ahead

#     # Create the figure
#     fig = go.Figure()
    
#     # Add training data
#     fig.add_trace(go.Scatter(
#         x=train_data.index,
#         y=train_data,
#         mode='lines',
#         name='Training Data',
#         line=dict(color='blue')
#     ))
    
#     # Add test data
#     fig.add_trace(go.Scatter(
#         x=test_data.index,
#         y=test_data,
#         mode='lines',
#         name='Test Data',
#         line=dict(color='green')
#     ))
    
#     # Add forecast
#     fig.add_trace(go.Scatter(
#         x=test_data.index,
#         y=forecast,
#         mode='lines',
#         name='Forecast',
#         line=dict(color='red')
#     ))
    
#     # Update layout with larger size and repositioned legend
#     fig.update_layout(
#         title=f'USD/MT Forecast for {selected_country}',
#         xaxis_title='Date',
#         yaxis_title='USD/MT Forecast',
#         legend=dict(x=0, y=1.1, orientation='h'),  # Horizontal legend above the chart
#         template='plotly',
#         width=1000,  # Set the width of the figure
#         height=600   # Set the height of the figure
#     )

#     st.plotly_chart(fig)

def forecast_ott_nc():

    if 'All' not in selected_out_turns:
        dftemp = df_new[df_new['Out Turn (Lbs)'].isin(selected_out_turns)]
    else:
        dftemp = df.copy()

    # Step 8: Filter data for selected nut counts if not 'All'
    if 'All' not in selected_nut_counts:
        dftemp = dftemp[dftemp['Nut Count'].isin(selected_nut_counts)]
    
    dftemp.reset_index(inplace=True)
    # Step 1: Calculate the top 10 countries by TOTAL USD
    top_countries = dftemp.groupby('FOREIGN COUNTRY')['TOTAL VALUE USD'].sum().nlargest(10).index
    
    # # Step 2: Allow user to select a country from the top 10
    # selected_country = st.selectbox('Choose a country', top_countries)  # Dropdown for country selection

    # # Step 2: Filter data for the selected country
    # df_country = df_new[df_new['FOREIGN COUNTRY'] == selected_country]

    # Step 2: Allow user to select multiple countries with a multiselect
    selected_countries = st.multiselect('Choose countries', ['All'] + list(top_countries), default='All')

    # Step 3: Filter data for the selected countries if not 'All'
    if 'All' in selected_countries:
        df_country = dftemp.copy()  # Don't filter if 'All' is selected
    else:
        df_country = dftemp[dftemp['FOREIGN COUNTRY'].isin(selected_countries)]

    df_country['DOMESTIC USDMT'].dropna(inplace=True)
    
    # Step 2: Aggregate or resample data over time (assuming 'Date' column exists)
    df_country['BILL DATE'] = pd.to_datetime(df_country['BILL DATE'])  # Ensure 'Date' is datetime
    df_country.set_index('BILL DATE', inplace=True)
    
    df_country['DOMESTIC USDMT'].fillna(0, inplace=True)
    # Resample data to monthly totals (you can choose weekly or daily based on your data)
    df_country_resampled = df_country['DOMESTIC USDMT'].resample('M').mean()
    df_country_resampled.dropna(inplace=True)
    # Step 3: Fit ARIMA model
    # Split into training and testing sets
    train_data = df_country_resampled[:-15]  # Use all but the last 12 months for training
    test_data = df_country_resampled[-15:]   # The last 12 months for testing

    try:
        
        # Fit ARIMA model (p, d, q values can be tuned)
        model = ARIMA(train_data, order=(7, 1, 9))
        model_fit = model.fit()
    
        # Step 4: Forecast future values
        forecast = model_fit.forecast(steps=15)  # Forecasting 12 months ahead
    
        # Create the figure
        fig = go.Figure()
        
        # Add training data
        fig.add_trace(go.Scatter(
            x=train_data.index,
            y=train_data,
            mode='lines',
            name='Training Data',
            line=dict(color='blue')
        ))
        
        # Add test data
        fig.add_trace(go.Scatter(
            x=test_data.index,
            y=test_data,
            mode='lines',
            name='Test Data',
            line=dict(color='green')
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=test_data.index,
            y=forecast,
            mode='lines',
            name='Forecast',
            line=dict(color='red')
        ))
        
        # Update layout with larger size and repositioned legend
        fig.update_layout(
            title=f'USD/MT Forecast for {selected_countries}, Out Turn {selected_out_turns}, Nut Counts {selected_nut_counts}',
            xaxis_title='Date',
            yaxis_title='USD/MT Forecast',
            legend=dict(x=0, y=1.1, orientation='h'),  # Horizontal legend above the chart
            template='plotly',
            width=1000,  # Set the width of the figure
            height=600   # Set the height of the figure
        )
    
        st.plotly_chart(fig)
    
    except Exception as e:
        st.error(f"Error in ARIMA model fitting: {e}")



# Function to extract 'Out Turn' and 'Nut Count' in any flexible format
def extract_quality_indicators(description):
    # Extract 'Out Turn' (case and space insensitive, allowing any text or characters in between)
    out_turn_match = re.search(r'(out\s*turn|otr|o\.?t\.?)\s*[:\-]?\s*.*?(\d+(\.\d+)?)', description, re.IGNORECASE)
    out_turn = out_turn_match.group(2) if out_turn_match else None
    
    # Extract 'Nut Count' or variations (case and space insensitive, allowing any text or characters in between)
    nut_count_match = re.search(r'(nut\s*count|nuts\s*count|nut\s*cnt|n\.?\s*c\.?|nut\s*coun9t)\s*[:\-]?\s*.*?(\d+(\.\d+)?)', description, re.IGNORECASE)
    nut_count = nut_count_match.group(2) if nut_count_match else None
    
    return pd.Series([out_turn, nut_count])

    # Create a CSV file from the DataFrame
def create_csv(df):
    return df.to_csv(index=False).encode('utf-8')

df = mts_conversion(df)
df = port_country_agg(df)

df['FINAL UNIT USD'] = df['TOTAL VALUE USD']/df['QUANTITY_MTS']
df['FINAL UNIT DOMESTIC'] = df['TOTAL VALUE DOMESTIC']/df['QUANTITY_MTS']
df['MANUAL USD'] = df['FINAL UNIT USD'] * df['EXCHNAGE RATE']
df['DOMESTIC USDMT'] = df['FINAL UNIT DOMESTIC']/df['EXCHNAGE RATE']

Q1 = df['FINAL UNIT USD'].quantile(0.25)
Q3 = df['FINAL UNIT USD'].quantile(0.75)
IQR = Q3 - Q1

# Define the outlier thresholds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the dataframe to remove outliers
df = df[(df['FINAL UNIT USD'] >= lower_bound) & (df['FINAL UNIT USD'] <= upper_bound)]

# Apply the function to extract quality indicators
df[['Out Turn (Lbs)', 'Nut Count']] = df['PRODUCT DESCRIPTION'].apply(extract_quality_indicators)

df[['Out Turn (Lbs)', 'Nut Count']] = df[['Out Turn (Lbs)', 'Nut Count']].astype(float)

df_new = df.dropna(subset=['Nut Count'])
df_new = df_new.dropna(subset=['Out Turn (Lbs)'])

n_ott = 5

# Get the top 'n' value counts and keep only those in the dataframe
top_n_ott = df_new['Out Turn (Lbs)'].round().value_counts().nlargest(n_ott)
top_n_values_ott = top_n_ott.index

n_nc = 5

# Get the top 'n' value counts and keep only those in the dataframe
top_n_nc = df_new['Nut Count'].round().value_counts().nlargest(n_nc)
top_n_values_nc = top_n_nc.index

df_new = df_new[df_new['Nut Count'].round().isin(top_n_values_nc)]
df_new = df_new[df_new['Out Turn (Lbs)'].round().isin(top_n_values_ott)]

df_new['Out Turn (Lbs)'] = df_new['Out Turn (Lbs)'].round().astype(int)
df_new['Nut Count'] = df_new['Nut Count'].round().astype(int)

 # Step 4: Get unique values for 'Out Turn (Lbs)' and 'Nut Count'
out_turns = df_new['Out Turn (Lbs)'].dropna().unique()
nut_counts = df_new['Nut Count'].dropna().unique()



# Streamlit App Layout
# st.title('Raw Cashew Nut (RCN) Trade Dashboard')
# Set page layout to wide
st.set_page_config(page_title="RCN Trade Dashboard", layout="wide")

st.write("Click the button below to download the source data as a CSV file.")
st.download_button(
        label="Download CSV",
        data=create_csv(df),
        file_name="data.csv",
        mime="text/csv",
        key="download_csv_button"  # Optional key for the button
    )

# Step 5: Allow user to select multiple 'Out Turn' values with multiselect
selected_out_turns = st.multiselect('Select Out Turn (Lbs)', ['All'] + sorted(out_turns), default='All')

# Step 6: Allow user to select multiple 'Nut Count' values with multiselect
selected_nut_counts = st.multiselect('Select Nut Count', ['All'] + sorted(nut_counts), default='All')


# Create Tabs for visualizations
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab9 = st.tabs(['Ports by Total USD', 'Ports by USD/MT', 'Flows by Total USD', 'Flows by USD/MT',\
                           'Monthly USD/MT by Country', 'Monthly USD by Country','Weekly Trend', 'Quarterly Forecast'])

with tab1:
    top_ports_by_total_usd()

with tab2:
    top_ports_by_usd_per_mt()

with tab3:
    flow_by_totalusd()

with tab4:
    flow_by_usd_per_mt()

with tab5:
    usd_per_mt_monthly()

with tab6:
    total_usd_monthly()

with tab7:
    weekly_moving_avg()

with tab9:
    forecast_ott_nc()

    
