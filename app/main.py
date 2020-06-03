# Data Handling
import pickle  # used for serializing and de-serializing a Python object
import numpy as np
from pydantic import BaseModel   # Pydantic is used to create objects in a model using BaseModel class
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import numpy
import onnxruntime as rt

# Server
import uvicorn   # provides support for http requests.
from fastapi import FastAPI # an API that the end-user can access.

# intialising the fastapi.
app = FastAPI()

with open('data/features.pickle','rb') as f:
    feature = pickle.load(f)
    print("features:",feature)
session = rt.InferenceSession("data/house.onnx")
first_input_name = session.get_inputs()[0].name
first_output_name = session.get_outputs()[0].name

# Creating objects i.e feature names
class Data(BaseModel):
    CRIM : float
    ZN : float
    INDUS : float
    CHAS : float
    NOX : float
    RM : float
    AGE : float
    DIS : float
    RAD : float
    TAX : float
    PTRATIO : float
    B : float
    LSTAT : float
    
@app.post("/predict")
def predict(data:Data):
    try:
        # Extract data in correct order
        data_dict = data.dict()
        to_predict = [data_dict[feature] for feature in feature]
        
        # dict to array
        
        to_predict = numpy.array(to_predict).reshape(1,-1)
        print("array:",to_predict)
        pred_onx = session.run([], {first_input_name: to_predict.astype(numpy.float32)})[0]
        return {"prediction":float(pred_onx[0])}
    except:
        return {"prediction": "error"}
