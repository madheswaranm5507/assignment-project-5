# SINGAPORE RESALE FLAT PRICES PREDICTING

# Introduction:

       This project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. 
       This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.

# Technologies Used:

       1. Python - This project is implemented using the Python programming language.
       
       2. Pandas - A Powerfull data manipulation in pandas. providing functionalities such as data filtering, dataframe create, transformation, and aggregation.
       
       3. Numpy - Is an essential library for numerical computations in Python, offering a vast array of functionalities to manipulate and operate on arrays and matrices efficiently.
       
       4. Scikit-Learn - This one of the most popular libraries for machine learning in Python. offering a wide range of supervised and unsupervised machine learning algorithms.
       
       5. Matplotlib - A wide range of plot types including line plots, scatter plots, bar plots, histograms, pie charts, and more. 
                       It also supports complex visualizations like 3D plots, contour plots, and image plots.
                       
       6. Seaborn - It provides a high-level interface for drawing attractive and informative statistical graphics.
       
       7. Pickle - A useful Python tool that allows to save the ML models, to minimise lengthy re-training and allow to share, commit, and re-load pre-trained machine learning models
       
       8. Streamlit - The user interface and visualization are created using the Streamlit framework.
       
       9. Hugging Face - This help to Deploy the machine learning models and user friendly interface in web cloud platfrom.


# Data Processing Methods:

      1. Data Collection and Preprocessing:
                Collect a dataset of resale flat transactions from the Singapore Housing and Development Board (HDB). Preprocess the data to clean and structure it for machine learning.
    
      2. Feature Engineering:
                Extract relevant features from the dataset, including town, flat type, storey range, floor area, flat model, and lease commence date.
                Create any additional features that may enhance prediction accuracy.
    
      3. Encoding and Data Type Conversion
                To prepare categorical features for modeling, used for Label Encoding and Ordinal Encoding. This technique transforms categorical values into numerical representations. 
                to convert data types to ensure they match the requirements of our modeling process.
    
      4. Handling the skewness using (Log Transfermation)
                Log transformation is reduce skewness in a distribution and make it more symmetric. primarily used to convert a skewed distribution to a normal distribution.
    
      5. Outliers Handling Using Interquartile Range(IQR)
                An outlier is a single data point that goes far outside the average value of a group. Outlier Removal in Dataset using Interquartile Range(IQR) Method Used
    
      6. Model Selection:
                Choose an appropriate machine learning model for Regression check with (LinearRegression,DecisionTreeRegressor,RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor) 
                RandomForestRegressor Algorithm is good accuracy in 95.5%, and without overfitting so will select algorithm.
    
      7. Model Evaluation:
                Evaluate the model's predictive performance using Regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE) and R2 Score.
    
      8. Streamlit Web Application:
                Develop a user-friendly web application using Streamlit that allows users to input details of a flat. Utilize the trained machine learning model to predict the resale price based on user inputs.
    
      9. Deployment on Hugging Face:
                Deploy the Streamlit application on the Hugging Face platform to make it accessible to users over the internet.

# Conclusion:
        This project offers precise for the resale prices of Singapore flats, along with insights into their availability, leveraging advanced machine learning methodologies. 
        users can engage with the model's predictions, empowering them to make well-informed decisions.
              

