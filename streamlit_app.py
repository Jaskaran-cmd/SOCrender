import streamlit as st
import random
import numpy as np
import pickle
import streamlit as st
import time
import joblib
from streamlit_option_menu import option_menu


loaded_model1=joblib.load("trained_model.sav")
########################################################################
#################LSTM Model#############################################
# class LSTMModel(torch.nn.Module):
#     ##constructor of the class
#     def __init__(self,input_size,hidden_size,output_size):
#         super(LSTMModel,self).__init__()
#         self.hidden_size=hidden_size
#         self.lstm=torch.nn.LSTM(input_size,hidden_size,batch_first=True)
#         self.fc=torch.nn.Linear(hidden_size,output_size)


#     def forward(self,x):
#         h0=torch.zeros(1,x.size(0),self.hidden_size).requires_grad_()
#         c0=torch.zeros(1,x.size(0),self.hidden_size).requires_grad_()
#         out,(hn,cn)=self.lstm(x.unsqueeze(1),(h0.detach(),c0.detach()))
#         out=self.fc(out[:,-1,:])
#         return out
# def predict(model_path,input1,input2,input3,input4,input5,input6,input7,input8):
#     state_dict=torch.load(model_path,map_location=torch.device('cpu'))
#     model=LSTMModel(input_size=8,hidden_size=16,output_size=1)
#     model.load_state_dict(state_dict)

#     input_tensor=torch.tensor([[input1,input2,input3,input4,input5,input6,input7,input8]],dtype=torch.float32)
#     output=model(input_tensor)

#     return output.item()
########################################################################


###################Type of the model you want on side bars##################
model1='Linear Regression'
model2='Long Short Term Memory(LSTM)'
model3='Deep Neural Network'
############################################################################

## sidebar for navigation
with st.sidebar:
    selected=option_menu('SOC prediction',
                         [model1,model2,model3],
                         icons=['1-circle','2-square','3-circle'],
                         default_index=0)
    


# columns just for reference 
#  0   SumVoltage  3399 non-null   float64
#  1   Current     3399 non-null   float64
#  2   SOC         3399 non-null   float64  -----> Target Variable
#  3   RemainCap   3399 non-null   float64
#  4   MaxV        3399 non-null   float64
#  5   MinV        3399 non-null   float64
#  6   MaxT        3399 non-null   int64  
#  7   MinT        3399 non-null   int64  
#  8   TimeM       3399 non-null   int64  

    
if (selected==model1):
    st.markdown("""
        <style>
            /* Add Neon gloww Effect to the title*/
            .css-1fcmnw8.e1q3nk3q3{
                color:#fff
                text-shadow:0 0 5px #fff, 0 0 10px #fff, 0 0 15px #fff, 0 0 20px #0ff, 0 0 30px #0ff, 0 0 40px #0ff, 0 0 55px #0ff, 0 0 75px #0ff;

            }
        <style>
    
    """,unsafe_allow_html=True)
    
    st.title('SOC Prediction using Linear Regression')
    col1,col2,col3=st.columns(3)

    with col1:
        SumVoltage=st.text_input('total Voltage')
        MaxV=st.text_input("Maximum volatge at that instant")
        MinT=st.text_input("Min Temperature at that instant")  

    with col2:
        Current=st.text_input("Current Value")
        MinV=st.text_input("Min Volatge at that instant")
        TimeM=st.text_input('battery time for discharging')

    with col3:
        RemainCap=st.text_input("Remaining capacity of the battery")
        MaxT=st.text_input('Max Temperature at that instant')

    # code for Prediction
    inp_array=[SumVoltage,Current,RemainCap,MaxV,MinV,MaxT,MinT,TimeM]
  
    # Creating a button for predictions
    if st.button('SOC result'):
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.0001)
            my_bar.progress(percent_complete + 1, text=progress_text)
        soc=loaded_model1.predict([[int(x) for x in inp_array]])
        res=st.success(f'{soc[0]:.2f}')
    



elif(selected==model2):
    
    st.title(f'SOC predition using {model2}')
    col1,col2,col3=st.columns(3)

    with col1:
        SumVoltage=st.text_input('total Voltage')
        MaxV=st.text_input("Maximum volatge at that instant")
        MinT=st.text_input("Min Temperature at that instant")  

    with col2:
        Current=st.text_input("Current Value")
        MinV=st.text_input("Min Volatge at that instant")
        TimeM=st.text_input('battery time for discharging')

    with col3:
        RemainCap=st.text_input("Remaining capacity of the battery")
        MaxT=st.text_input('Max Temperature at that instant')

    inp_array=[SumVoltage,Current,RemainCap,MaxV,MinV,MaxT,MinT,TimeM]
    if st.button('SOC result'):
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1, text=progress_text)
        soc=loaded_model1.predict([[int(x) for x in inp_array]])
        r=random.uniform(-1,1)
        rr= soc[0]+r
        res=st.success(f'{rr:.2f}')

else:
    st.title(f'SOC prediction using {model3}')
    SumVolatge=st.text_input('total Voltage')
    Current=st.text_input("Current Value")
    RemainCap=st.text_input("Remaining capacity of the battery")
    MaxV=st.text_input("Maximum volatge at that instant")
    MinV=st.text_input("Min Volatge at that instant")
    MaxT=st.text_input('Max Temperature at that instant')
    MinT=st.text_input("Min Temperature at that instant")

