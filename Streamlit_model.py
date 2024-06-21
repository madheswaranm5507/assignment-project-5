# import packages
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
from PIL import Image

# user input option
class option:
    year_values = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

    town_values = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
                    'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
                    'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
                    'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
                    'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
                    'TOA PAYOH', 'WOODLANDS', 'YISHUN']
    
    town_dict = {'ANG MO KIO':0, 'BEDOK':1, 'BISHAN':2, 'BUKIT BATOK':3, 'BUKIT MERAH':4,
                    'BUKIT PANJANG':5, 'BUKIT TIMAH':6, 'CENTRAL AREA':7, 'CHOA CHU KANG':8,
                    'CLEMENTI':9, 'GEYLANG':10, 'HOUGANG':11, 'JURONG EAST':12, 'JURONG WEST':13,
                    'KALLANG/WHAMPOA':14, 'MARINE PARADE':15, 'PASIR RIS':16, 'PUNGGOL':17,
                    'QUEENSTOWN':18, 'SEMBAWANG':19, 'SENGKANG':20, 'SERANGOON':21, 'TAMPINES':22,
                    'TOA PAYOH':23, 'WOODLANDS':24, 'YISHUN':25}
    

    flat_type_values = ['3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM', 'EXECUTIVE', '1 ROOM',
                        'MULTI-GENERATION']
    
    flat_type_dict = {'3 ROOM':2, '4 ROOM':3, '5 ROOM':4, '2 ROOM':1, 'EXECUTIVE':5, '1 ROOM':0,
                      'MULTI-GENERATION':6}
    
    
    flat_model_values = ['Improved', 'New Generation', 'Model A', 'Standard', 'Simplified',
                        'Premium Apartment', 'Maisonette', 'Apartment', 'Model A2',
                        'Type S1', 'Type S2', 'Adjoined flat', 'Terrace', 'DBSS',
                        'Model A-Maisonette', 'Premium Maisonette', 'Multi Generation',
                        'Premium Apartment Loft', 'Improved-Maisonette', '2-room', '3Gen']
    
    flat_model_dict = {'Improved':5, 'New Generation':12, 'Model A':8, 'Standard':17, 'Simplified':16,
                        'Premium Apartment':13, 'Maisonette':7, 'Apartment':3, 'Model A2':10,
                        'Type S1':19, 'Type S2':20, 'Adjoined flat':2, 'Terrace':18, 'DBSS':4,
                        'Model A-Maisonette':9, 'Premium Maisonette':15, 'Multi Generation':11,
                        'Premium Apartment Loft':14, 'Improved-Maisonette':6, '2-room':0, '3Gen':1}
    

    storey_start_values = [0.01185826, 1.38629436, 1.94591015, 2.30258509, 2.56494936, 2.77258872, 
                           2.94443898, 3.09104245, 3.21887582, 3.33220451, 3.4339872, 3.52636052, 
                           3.61091791, 3.67702119]
    
    storey_end_values = [1.09861229, 1.79175947, 2.19722458, 2.48490665, 2.7080502, 2.89037176, 
                         3.04452244, 3.17805383, 3.29583687, 3.40119738, 3.49650756, 3.52462742]
    

    floor_area_sqm = [37.5, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 
                     52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 60.3, 61.0, 62.0, 63.0, 63.1, 64.0, 
                     65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 
                     81.0, 82.0, 83.0, 83.1, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 
                     96.0, 97.0, 98.0, 99.0, 100.0, 100.2, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 
                     110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 
                     125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0, 132.0, 133.0, 134.0, 135.0, 136.0, 137.0, 138.0, 139.0, 
                     140.0, 141.0, 142.0, 143.0, 144.0, 145.0, 146.0, 147.0, 148.0, 149.0, 150.0, 151.0, 152.0, 153.0, 153.5]
    
    remaining_lease_month = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    remaining_lease_year = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 
                            66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 
                            91, 92, 93, 94, 95, 96, 97]
    
    lease_commence_date = [1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 
                           1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 
                           2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    

# Functions in predicts the "resale prices"

def resale_flat_prices_predict(year, town, flat_type, floor_area_sqm, flat_model,
                        storey_start, storey_end, remaining_lease_year,
                        remaining_lease_month, lease_commence_date):
    
    # pickle file for Regression Model
    with open("Resale Flat Prices Model Final.pkl", "rb") as f:
        model_regression = pickle.load(f)

        user_regression_data = np.array([[year, option.town_dict[town], option.flat_type_dict[flat_type], floor_area_sqm, option.flat_model_dict[flat_model],
                                        storey_start, storey_end, remaining_lease_year, remaining_lease_month, lease_commence_date]])
                                    
        
        y_pred = model_regression.predict(user_regression_data)

        y_pred_exponential_value = np.exp(y_pred[0])

        return y_pred_exponential_value
    

# Streamlit part
st.set_page_config(layout="wide")
st.title(":red[SINGAPORE RESALE FLAT PRICES PREDICTING]")

with st.sidebar:
    select = option_menu("DATA EXPLORATION", options=["HOME", "FLAT PRICES PREDICTION", "DATA PROCESSING METHODS"])

if select == "HOME":
    image = Image.open(r"D:\DS PROJECTS\Singapore  Resale Flat Prices Predicting\beautiful-architecture-building-exterior-city-kuala-lumpur-skyline.jpg")
    st.image(image, width=1000)

    st.write("")
    st.header(":green[Introduction]")
    st.write("""This project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. 
             This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.""")

    st.header(":green[Technologies Used]")
    st.write("1. Python - This project is implemented using the Python programming language.")
    st.write("2. Pandas - A Powerfull data manipulation in pandas. providing functionalities such as data filtering, dataframe create, transformation, and aggregation.")
    st.write("3. Numpy - Is an essential library for numerical computations in Python, offering a vast array of functionalities to manipulate and operate on arrays and matrices efficiently.")
    st.write("4. Scikit-Learn - This one of the most popular libraries for machine learning in Python. offering a wide range of supervised and unsupervised machine learning algorithms.")
    st.write("5. Matplotlib - A wide range of plot types including line plots, scatter plots, bar plots, histograms, pie charts, and more. It also supports complex visualizations like 3D plots, contour plots, and image plots.")
    st.write("6. Seaborn - It provides a high-level interface for drawing attractive and informative statistical graphics.")
    st.write("7. Pickle - A useful Python tool that allows to save the ML models, to minimise lengthy re-training and allow to share, commit, and re-load pre-trained machine learning models")
    st.write("8. Streamlit - The user interface and visualization are created using the Streamlit framework.")
    st.write("9. Hugging Face - This help to Deploy the machine learning models and user friendly interface in web cloud platfrom.")


if select == "FLAT PRICES PREDICTION":

    st.header(":green[RESALE FLAT PRICES PREDICTION]")
    st.write("")

    col1,col2 = st.columns(2)
    with col1:
        year = st.selectbox(label="Year", options=option.year_values)

        town = st.selectbox(label="Town", options=option.town_values)

        flat_type = st.selectbox(label="Flat_Type", options=option.flat_type_values)

        floor_area_sqm = st.selectbox(label="Floor_Area_sqm", options=option.floor_area_sqm)

        flat_model = st.selectbox(label="Flat_Model", options=option.flat_model_values)

    with col2:
        storey_start = st.selectbox(label="Storey_Start", options=option.storey_start_values)

        storey_end = st.selectbox(label="Storey_End", options=option.storey_end_values)   

        remaining_lease_year = st.selectbox(label="Remaining_Lease_Year", options=option.remaining_lease_year) 

        remaining_lease_month = st.selectbox(label="Remaining_Lease_Month", options=option.remaining_lease_month)

        lease_commence_date = st.selectbox(label="Lease_Commence_Date", options=option.lease_commence_date)

    button = st.button(":blue[PREDICT THE RESALE FLAT PRICE]", use_container_width=True)

    if button:
        predict_price = resale_flat_prices_predict(year, town, flat_type, floor_area_sqm, flat_model, storey_start, storey_end,
                                                   remaining_lease_year, remaining_lease_month, lease_commence_date )
        
        st.write(":red[THE RESALE FLAT PRICE IS : ]", predict_price)


if select == "DATA PROCESSING METHODS":

    st.header(":green[Data Collection and Preprocessing:]")
    st.write("Collect a dataset of resale flat transactions from the Singapore Housing and Development Board (HDB). Preprocess the data to clean and structure it for machine learning.")

    st.header(":green[Feature Engineering:]")
    st.write("Extract relevant features from the dataset, including town, flat type, storey range, floor area, flat model, and lease commence date. Create any additional features that may enhance prediction accuracy.")
    
    st.header(":green[Encoding and Data Type Conversion]")
    st.write("""To prepare categorical features for modeling, used for Label Encoding and Ordinal Encoding. This technique transforms categorical values into numerical representations. 
              to convert data types to ensure they match the requirements of our modeling process.""")
    
    st.header(":green[Handling the skewness using (Log Transfermation)]")
    st.write("""Log transformation is reduce skewness in a distribution and make it more symmetric. primarily used to convert a skewed distribution to a normal distribution.""")

    st.header(":green[Outliers Handling Using Interquartile Range(IQR)]")
    st.write("An outlier is a single data point that goes far outside the average value of a group. Outlier Removal in Dataset using Interquartile Range(IQR) Method Used ")

    st.header(":green[Model Selection:]")
    st.write("""Choose an appropriate machine learning model for Regression check with (LinearRegression,DecisionTreeRegressor,RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor) 
                RandomForestRegressor Algorithm is good accuracy in 95.5%, and without overfitting so will select algorithm.""")

    st.header(":green[Model Evaluation:]")
    st.write("Evaluate the model's predictive performance using Regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE) and R2 Score.")

    st.header(":green[Streamlit Web Application:]")
    st.write("Develop a user-friendly web application using Streamlit that allows users to input details of a flat. Utilize the trained machine learning model to predict the resale price based on user inputs.")

    st.header(":green[Deployment on Hugging Face:]")
    st.write("Deploy the Streamlit application on the Hugging Face platform to make it accessible to users over the internet.")
    
    

    st.write("")
    st.write("")
    st.header(":green[Conclusion:]")
    st.write("""This project offers precise for the resale prices of Singapore flats, along with insights into their availability, 
                leveraging advanced machine learning methodologies. users can engage with the model's predictions, 
                empowering them to make well-informed decisions.""")    
    

