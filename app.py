import streamlit as st
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.model import model_from_json
from tensorflow.python.keras.preprocessing.sequence import pad_sequence
from PIL import Image

#import pre-trained model
json_file=open('./model.json','r')
loaded_json_model = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_json_model)

#load weights

loaded_model.load_weight("/.model.h5")

with open('./bart-chalkboard-data.txt','r',encoding='utf-8') as file:
    data = file.read()


