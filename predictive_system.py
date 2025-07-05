import numpy as np
import pickle

loaded_model=pickle.load(open("trained_model.sav", 'rb'))

with open("scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)
    
input_data=(4,110,92,0,0,37.6,0.191,30)
to_np_array=np.asarray(input_data)

reshaped_array=to_np_array.reshape(1,-1)
std_data=scaler.transform(reshaped_array)
print(std_data)

prediction=loaded_model.predict(reshaped_array)
print(prediction)

if prediction[0]==0:
  print('This person is not diabetic')
else:
  print('This person is diabetic')
