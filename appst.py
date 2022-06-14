from PIL import Image
import numpy as np
import pickle as pkl
import streamlit as st
from streamlit_drawable_canvas import st_canvas

#importing model
model = pkl.load(open("model.pkl", 'rb'))

#fromatting image data
def format_image(arr):
    '''takes image array as input and returns trainable array'''
    im = Image.fromarray(arr).convert('L')
    im = im.resize(size = (28,28))
    #im.show()
    arr = np.array(im).reshape(1,-1)
    print(arr)
    return arr

#predicting the MNIST classifier
def predict(img = None):
    '''creates an st.write component and prints the predicted result'''
    imarr = format_image(img)
    #print(arr.shape)
    with col2:
            st.write("**predicted result**")

            st.write(f"### {model.predict(imarr)[0]}")
    
#APP Layout
st.title("MNIST Classifier")
st.header("Handwritten Digit classification")

col1, col2 = st.columns([3,1])
with col2:
    c = st.empty()
    c.write("Draw doodle and predict")

with col1:
# Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=30,
        stroke_color='#FFFFFF',
        background_color='#000000',
        update_streamlit=True,
        height=500,
        width = 500,
        drawing_mode='freedraw',
        key="canvas",
)

st.button(label = 'try', on_click = predict, kwargs = {'img' :canvas_result.image_data})

#END
