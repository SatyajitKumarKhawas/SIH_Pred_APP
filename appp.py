import streamlit as st
import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA

# Ignore warnings
warnings.filterwarnings('ignore')
background_image_css = """
<style>
    .stApp {
        background-image: url('https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.freepik.com%2Ffree-photos-vectors%2Fdark-forest-background&psig=AOvVaw0NlZXyOJWoRUOCjcaikDsH&ust=1725778761453000&source=images&cd=vfe&opi=89978449&ved=2ahUKEwjv2MiBobCIAxUla2wGHRqpJ7EQjRx6BAgAEBg');  /* Replace with your image URL */
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
</style>
"""

# Apply the CSS
st.markdown(background_image_css, unsafe_allow_html=True)
# CSS to add background image
background_image_css = """
<style>
    .stApp {
        background-image: url('https://static.vecteezy.com/system/resources/previews/027/004/037/non_2x/green-natural-leaves-background-free-photo.jpg');  /* Replace with your image URL */
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
</style>
"""

# Inject the CSS into the Streamlit app
st.markdown(background_image_css, unsafe_allow_html=True)




# Streamlit app title and description
st.title("AGRI-FORECAST")
# Display an image with specific width
#st.image("NPIC-2023811163621.jpg", caption="Resized Image", width=450)  # Width is in pixels

st.header("Predict Future Values of Agri-Horticulture Commodities")
st.subheader("Enter the commodity,month and year for which you need predictions, and get the predicted average sales for that month.")
com = st.selectbox(
    "COMMODITY", 
    ["Gram Dal","Rice", "Wheat", "Atta", "Tur/Arhar Dal", "Urad Dal", "Moong Dal", 
     "Masoor Dal", "Sugar", "Gur", "Groundnut Oil", "Mustard Oil", "Vanaspati", 
     "Sunflower Oil", "Soya Oil", "Palm Oil", "Tea", "Milk", "Salt", "Potato", 
     "Onion", "Tomato"]
)


if com == "Gram Dal":
    # Load and cache the data once
    @st.cache_data
    def load_data(file_path):
        df = pd.read_csv(file_path)
        df.columns = ['Month', 'Sales']
        df.dropna(inplace=True)
        df['Month'] = pd.to_datetime(df['Month'])
        df.set_index('Month', inplace=True)
        return df

    # Train and cache the ARIMA model
    @st.cache_resource
    def train_arima_model(df, order):
        model = ARIMA(df['Sales'], order=order)
        model_fit = model.fit()
        return model_fit

    # File path for internal dataset
    file_path = "c:/Users/satya/OneDrive/Desktop/PANDAS/Dal_Price.csv"

    try:
        # Load the data
        df = load_data(file_path)

        # Define ARIMA model order
        order = (1, 3, 1)

        # Train the ARIMA model
        model_fit = train_arima_model(df, order)

        # Forecasting for 12 months (1 year ahead)
        forecast12 = model_fit.get_forecast(steps=120)
        forecast_values_12 = forecast12.predicted_mean

        # Create a date range for the forecasted values with 'MS' frequency (start of the month)
        forecast_dates12 = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=120, freq='MS')
        forecast_values_12.index = forecast_dates12

        # Combine historical and forecast data
        combined_series = pd.concat([df['Sales'], forecast_values_12])

        # Allow the user to input past or future dates
        st.subheader('Get Prediction or Historical Value for a Specific Month and Year')

        # Allow user to select any year and month within the range of historical and forecast data
        min_year = combined_series.index.min().year
        max_year = combined_series.index.max().year
        year = st.number_input("Enter the Year", min_value=min_year, max_value=max_year, value=min_year)
        month = st.selectbox("Enter the Month", list(range(1, 13)), index=0)

        # Add a button to get the prediction or historical value
        if st.button("Get Value"):
            # Construct the date string for lookup
            date = f"{year}-{month:02d}-01"
            
            # Show the value for the selected date
            try:
                value = combined_series.loc[date]
                if pd.Timestamp(date) in df.index:
                    st.subheader(f"Historical Sales for {date}: {value:.2f}")
                else:
                    st.subheader(f"Predicted Sales for {date}: {value:.2f}")
            except KeyError:
                st.error("The selected date is out of the available range.")
            st.image("download.png")

    except FileNotFoundError:
        st.error("The specified file was not found. Please check the file path and try again.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
if com !="Gram Dal":
    st.text("Model is in progress......")
