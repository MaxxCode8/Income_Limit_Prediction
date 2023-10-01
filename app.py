# So have to build Web App using Streamlit
# for this Income limit prediction project 
# i have trained XGBoost model and saved it as joblib file
# Input Features are: ['age', 'working_week_per_year', 'gains', 'education', 'gender', 'occupation_code', 
# 'stocks_status', 'industry_code_main', 'total_employed', 'tax_status', 'household_stat','losses']
# age                         int64  min>0 max>100
# working_week_per_year       int64  min>0 max>52
# gains                       int64  min>0 max>99999
# education                category  
# gender                   category
# occupation_code             int64
# stocks_status               int64
# industry_code_main       category
# total_employed              int64
# tax_status               category
# household_stat           category
# losses                      int64
# So using streamlit ui features i have to take in above inputs and predict the income limit
# 
import streamlit as st
import joblib
import pandas as pd 
from utilities import ordinal_encoder, target_encoder, scale_features

# loading the trained model
model = joblib.load('Model/xgboost_tunned_81_v6.joblib')


st.title('Income Limit Prediction')

opt_edu = ['High school graduate', 'Children', 'Some college but no degree', 'Bachelors degree(BA AB BS)', '7th and 8th grade', '10th grade', '11th grade', 'Masters degree(MA MS MEng MEd MSW MBA)', '9th grade', 'Associates degree-occup /vocational', 'Associates degree-academic program', '5th or 6th grade', '12th grade no diploma', '1st 2nd 3rd or 4th grade', 'Prof school degree (MD DDS DVM LLB JD)', 'Doctorate degree(PhD EdD)', 'Less than 1st grade']
opt_gender = ['Male', 'Female']
opt_industry_code = ['Not in universe or children', 'Retail trade', 'Manufacturing-durable goods', 'Education', 'Manufacturing-nondurable goods', 'Construction', 'Finance insurance and real estate', 'Business and repair services', 'Medical except hospital', 'Public administration', 'Other professional services', 'Transportation', 'Hospital services', 'Wholesale trade', 'Agriculture', 'Personal services except private HH', 'Social services', 'Entertainment', 'Communications', 'Utilities and sanitary services', 'Private household services', 'Mining', 'Forestry and fisheries', 'Armed Forces']
opt_tax_status = ['Nonfiler', 'Joint both under 65', 'Single', 'Joint both 65+', 'Head of household', 'Joint one under 65 & one 65+']
opt_household = ['Householder', 'Child <18 never marr not in subfamily', 'Spouse of householder', 'Nonfamily householder', 'Child 18+ never marr Not in a subfamily', 'Secondary individual', 'Other Rel 18+ ever marr not in subfamily', 'Grandchild <18 never marr child of subfamily RP', 'Other Rel 18+ never marr not in subfamily', 'Grandchild <18 never marr not in subfamily', 'Child 18+ ever marr Not in a subfamily', 'Child under 18 of RP of unrel subfamily', 'RP of unrelated subfamily', 'Other Rel 18+ spouse of subfamily RP', 'Child 18+ ever marr RP of subfamily', 'Other Rel 18+ ever marr RP of subfamily', 'Other Rel <18 never marr child of subfamily RP', 'Other Rel <18 never marr not in subfamily', 'Child 18+ never marr RP of subfamily', 'Grandchild 18+ never marr not in subfamily', 'In group quarters', 'Child 18+ spouse of subfamily RP', 'Other Rel 18+ never marr RP of subfamily', 'Child <18 never marr RP of subfamily', 'Spouse of RP of unrelated subfamily', 'Grandchild 18+ ever marr not in subfamily', 'Child <18 ever marr not in subfamily', 'Grandchild 18+ spouse of subfamily RP', 'Child <18 ever marr RP of subfamily', 'Grandchild 18+ ever marr RP of subfamily', 'Other Rel <18 ever marr RP of subfamily', 'Grandchild 18+ never marr RP of subfamily', 'Other Rel <18 ever marr not in subfamily', 'Other Rel <18 never married RP of subfamily', 'Other Rel <18 spouse of subfamily RP', 'Child <18 spouse of subfamily RP', 'Grandchild <18 ever marr not in subfamily', 'Grandchild <18 never marr RP of subfamily']

def main():
    # Categorical input features
    col1, col2, col3 = st.columns(3)

    with col1:
        education = st.selectbox('Education:', options=opt_edu)
        gender = st.selectbox('Gender:', options=opt_gender)

    with col2:
        industry_code_main = st.selectbox('Industry code main:', options=opt_industry_code)
        tax_status = st.selectbox('Tax status:', options=opt_tax_status)

    with col3:
        household_stat = st.selectbox('Household stat:', options=opt_household)

    # Numerical input features
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider('Age:', 0, 100)
        working_week_per_year = st.slider('Working week per year:', 0, 52)
        gains = st.slider('Gains:', 0, 99999)

    with col2:
        occupation_code = st.slider('Occupation code:', 0, 46)
        stocks_status = st.slider('Stocks status:', 0, 99999)
        total_employed = st.slider('Total employed:', 0, 6)
        losses = st.slider('Losses:', 0, 4608)

    submit = st.button('Submit')
    
    if submit:
        
        inputs = {'age': age, 
                'working_week_per_year' : working_week_per_year, 
                'gains' : gains, 
                'occupation_code' : occupation_code,
                'stocks_status' : stocks_status, 
                'total_employed' : total_employed, 
                'losses' : losses, 
                'gender' : gender, 
                'tax_status' : tax_status,
                'education' : education, 
                'industry_code_main' : industry_code_main, 
                'household_stat' : household_stat
                }


        # inp dataframe :::: 
        inpdf = pd.DataFrame(inputs, index=[0])

        obj_cols = inpdf.select_dtypes("object").columns
        inpdf[obj_cols] = inpdf[obj_cols].astype("category")

        # Encoding ... 
        inpdf = pd.concat([inpdf[['age', 'working_week_per_year', 'gains', 'occupation_code',
            'stocks_status', 'total_employed', 'losses']],ordinal_encoder(inpdf), target_encoder(inpdf)], axis=1)
        
        print(inpdf)
        
        # Scale the new input features
        inpdf = scale_features(inpdf)
        print()
        print(inpdf)
        print()

        # Make a prediction
        prediction = model.predict(inpdf)
        if prediction == 0:
            pred = 'Below limit'
        else:
            pred = 'Above limit'
        print(prediction)    
        # Display the prediction
        st.write(f'Prediction: {pred}')
        
if __name__ == '__main__':
    main()