import streamlit as st
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.model import model_from_json
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
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

#creating the text generator function
def generate_text(model,tokenizer,max_length,seed_text,n_words):
    text_generated=seed_text
    for i in range(n_words):
        encoded = tokenizer.texts_to_sequences([text_generated])[0]
        encoded = pad_sequences([encoded],max_len=max_length,padding= 'pre')
        yhat=model.predict_classes(encoded,verbose=0)
        
        predict_word=''
        for word,index in tokenizer.word_index.items():
            if index ==yhat :
                predict_word= word
                break
        text_generated+=' ' + predict_word
    return text_generated

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

max_length = 14


#using streamlit to create a streamlit app UI

st.title("The Simpsons Chalkboard Gag Text Generator")

image= Image.open('./1.jpg')
st.image(image,use_column_width=True)

n_words =st.number_input('Type the number of words you want to generate')
seed_text =st.text_input("Type a word or words you want to generate after ")
if n_words and seed_text:
    st.header(generate_text(loaded_json_model,tokenizer,max_length-1,seed_text,n_words))
else:
    st.warning("Please input a word and a number")
    