import joblib
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from streamlit_option_menu import option_menu
import tensorflow as tf
import numpy as np
import pandas as pd


def preprocessing(df):

    df["checkin_date"] = pd.to_datetime(df['checkin_date'])
    df["checkout_date"] = pd.to_datetime(df['checkout_date'])
    df["booking_date"] = pd.to_datetime(df['booking_date'])
    
    df["checkin_week"] = df['checkin_date'].dt.isocalendar().week
    df["checkin_month"] = df['checkin_date'].dt.month
    df["checkin_year"] = df['checkin_date'].dt.year

    df["checkout_week"] = df['checkout_date'].dt.isocalendar().week
    df["checkout_month"] = df['checkout_date'].dt.month

    df["advance_booking"] = (df['checkin_date'] - df["booking_date"]).dt.days
    df["days_stayed"] = (df['checkout_date'] - df['checkin_date']).dt.days

    df["weekdays_stayed"] = np.busday_count(df['checkin_date'].values.astype('datetime64[D]'),
                                                df['checkout_date'].values.astype('datetime64[D]'))

    df["weekends_stayed"] = df["days_stayed"] - df["weekdays_stayed"]

    df["dropped_days"] = df["roomnights"] - df["days_stayed"]

    df["total_pax_days"] = df['days_stayed'] * df['total_pax']

    calc_mean = df.groupby(['resort_id'], axis=0).agg(
        {"total_pax":"mean","days_stayed":"mean","advance_booking":"mean","total_pax_days":"mean"}).reset_index()
    calc_mean.columns = ['resort_id','totalpax_mean',"days_stayed_resmean","advance_booking_resmean","totpaxdays_resmean"]
    df = df.merge(calc_mean,on=['resort_id'],how='left')

    calc_mean = df.groupby(['memberid'], axis=0).agg(
        {"total_pax":"mean","days_stayed":"mean","advance_booking":"mean","reservation_id":"count",
         "roomnights":"mean","numberofadults":"mean","numberofchildren":"mean","weekends_stayed":"mean", "weekdays_stayed":"mean",
         "total_pax_days":"mean"}).reset_index()
    calc_mean.columns = ['memberid','totalpax_memmean',"days_stayed_memmean","advance_booking_memmean",
                         "res_memcnt","roomnights_memmean","adults_memmean","child_memmean","weekends_memmean","weekdays_memmean",
                         "totpaxdays_memmean"]
    df = df.merge(calc_mean,on=['memberid'],how='left')

    calc_mean = df.groupby(['memberid'], axis=0).agg(
        {"days_stayed":"sum","advance_booking":"sum","total_pax":"sum","numberofadults":"sum","numberofchildren":"sum",
         "weekends_stayed":"sum","total_pax_days":"sum"}).reset_index()
    calc_mean.columns = ['memberid',"days_stayed_memsum","advance_booking_memsum","total_pax_memsum","adults_memsum","child_memsum",
                         "weekend_memsum","totpaxdays_memsum"]
    df = df.merge(calc_mean,on=['memberid'],how='left')

    calc_mean = df.groupby(['memberid','resort_id'], axis=0).agg(
        {"total_pax":"mean","days_stayed":"mean","roomnights":"mean","reservation_id":"count","total_pax_days":"mean"}).reset_index()
    calc_mean.columns = ['memberid','resort_id','totalpax_memresmean',"days_stayed_memresmean","roomnights_memresmean",
                         "book_memrescnt","totpaxdays_memresmean"]
    df = df.merge(calc_mean,on=['memberid','resort_id'],how='left')

    df["passengers_dropped"] = df["total_pax"] - (df["numberofadults"] + df["numberofchildren"])

    df = df.drop(df[df['roomnights'] < 0].index)
    df = df.drop(df[df['advance_booking'] < 0].index)
    df = df.drop(df[df['days_stayed'] < 0].index)
    df = df.drop(df[(df["numberofadults"] + df["numberofchildren"]) == 0].index)
    df = df.drop(df[df['total_pax'] == 0].index)

    df = df.dropna()

    reservation_id=df["reservation_id"]

    drop = ['checkin_date','checkout_date','booking_date','memberid','reservation_id']
    df = df.drop(drop, axis=1)

    categorical_vars =df.select_dtypes(include=['object']).columns.tolist()

    for i in categorical_vars:
            lbl = LabelEncoder()
            lbl.fit(list(df[i].values))
            df[i] = lbl.transform(list(df[i].values))

    for var in categorical_vars:
        df[var] = df[var].astype('category')

    df = df.drop(['booking_type_code', 'book_memrescnt', 'child_memmean', 'child_memsum', 'weekends_stayed', 'res_memcnt', 'reservationstatusid_code'],axis =1)

    return df,reservation_id



def predict(X):

    X=X.values
    cat_model = joblib.load('models/cat_model.pkl')
    lgb_model = joblib.load('models/lgb_model.pkl')
    grad_model = joblib.load('models/grad_model.pkl')

    cat_pred = cat_model.predict(X)
    grad_pred = grad_model.predict(X)
    lgb_pred = lgb_model.predict(X)

    sum = grad_pred + cat_pred + lgb_pred
    Y_pred = sum / 3
    
    return Y_pred




with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

app_mode = option_menu(
    menu_title=None,
    options=["Home", "Prediction"],
    icons=["house-door", "graph-up-arrow"],
    orientation="horizontal",
    styles={
        "container": {
                "padding": "0!important",
        },
        "icon": {
            "font-size": "20px",
        },
        "nav-link": {
            "font-size": "20px",
            "margin": "0px",
            "padding": "7px 0 7px 0",
        },
        "nav-link-selected": {
            "font-weight": "100",
        }
    }
)

# Home Page
if app_mode == "Home":
   pass
# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)
        st.write("Uploaded CSV data:")
        st.write(df)
        
        df,reservation_id = preprocessing(df)
        predicted_data = predict(df)
        new_df = pd.DataFrame({"reservation_id":reservation_id,"answer":predicted_data})
        st.write("Prerdicted data:")
        st.write(new_df)
        csv = new_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='data.csv',
            mime='text/csv'
        )
