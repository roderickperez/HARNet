import streamlit as st
import matplotlib.pyplot as plt
############################
import train as train
import predict as predict
############################

plt.rcParams["figure.figsize"] = (6, 2)
plt.rcParams.update({'font.size': 6})
##############################
st.set_page_config(page_title="HARNet | UniWien Research Project",
                   page_icon=":chart_with_upwards_trend:",
                   layout="wide")

#########################
# ---- Header ----
with st.container():
    st.title(':chart_with_upwards_trend: HarNet App')


#####################################
# Define pages based on apps imported.
PAGES = {
    "Train": train,
    "Predict": predict
}

st.sidebar.image("images/uniWienLogo.png", use_column_width=True)

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Select Mode: ", list(PAGES.keys()))
page = PAGES[selection]
page.app()
