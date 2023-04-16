import torch
import torch.nn as nn
from flask import Flask,request
import json

app=Flask(__name__)

class LSTMModel(nn.Module):
    ##constructor of the class
    def __init__(self,input_size,hidden_size,output_size):
        super(LSTMModel,self).__init__()
        self.hidden_size=hidden_size
        self.lstm=nn.LSTM(input_size,hidden_size,batch_first=True)
        self.fc=nn.Linear(hidden_size,output_size)


    def forward(self,x):
        h0=torch.zeros(1,x.size(0),self.hidden_size).requires_grad_()
        c0=torch.zeros(1,x.size(0),self.hidden_size).requires_grad_()
        out,(hn,cn)=self.lstm(x.unsqueeze(1),(h0.detach(),c0.detach()))
        out=self.fc(out[:,-1,:])
        return out
        

def predict_fn(model_path,lolinput):


    state_dict=torch.load(model_path,map_location=torch.device('cpu'))
    model=LSTMModel(input_size=8,hidden_size=16,output_size=1)
    model.load_state_dict(state_dict)

    input_tensor=torch.tensor(lolinput,dtype=torch.float32)
    output=model(input_tensor)

    return output.item()

@app.route('/predict',methods=['POST'])
def predict():
    data=json.loads(request.data.decode())
    input_data=data['input_data']
    output_data=predict('LSTMmodel.pt',input_data)
    return json.dumps({'output_data':output_data})


if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000)