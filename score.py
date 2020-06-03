# Data Handling
import pickle  # used for serializing and de-serializing a Python object
import numpy as np
from pydantic import BaseModel   # Pydantic is used to create objects in a model using BaseModel class
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import numpy
import onnxruntime as rt
import os
import json


# Called when the deployed service starts
def init():
    global model
    global feature
    global first_input_name
    global first_output_name
    global session

    # Get the path where the deployed model can be found.
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), './models')
    # load models
    #model = load_model(model_path + '/model.pickle')
    
    with open(model_path + '/features.pickle','rb') as f:
        feature = pickle.load(f)
        #print("features:",feature)

    session = rt.InferenceSession(model_path + '/house.onnx')
    first_input_name = session.get_inputs()[0].name
#print("input:",first_input_name)
    first_output_name = session.get_outputs()[0].name
    
#print("output:",first_output_name)

# Handle requests to the service
def run(data):
    try:
        # Extract data in correct order
        data_dict = json.loads(data)
        
        
        data_dict = numpy.array(data_dict).reshape(1,-1)
        #print("array:",data_dict)
        pred_onx = session.run([first_output_name], {first_input_name: data_dict.astype(numpy.float32)})[0]
        
        #print("prediction",pred_onx)
        
        return {"prediction":float(pred_onx[0])}
    except:
        return {"prediction": "error"}
