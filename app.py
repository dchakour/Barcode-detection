import streamlit as st
from model.tiny_yolo import TinyYolo
from barcode import get_barcode

def main():
    st.title("Barcode detection app")
    image_file = st.file_uploader("Please upload your image:",
                                    type=['jpg' ,'jpeg', 'png'])
    model = TinyYolo()
    
    if image_file is not None:
        imageText = st.empty()
        imageLocation = st.empty()
        imageText.text("Image uploaded")
        imageLocation.image(image_file)
        image_predicted = model.predict(image_file)
        
        if st.button("Launch barcode detection"):
            
            if image_predicted is not None:
                imageText.text("Image with barcode detected")
                imageLocation.image(image_predicted)
                barcode, info = get_barcode(image_file)
                
                if barcode is not None:
                    st.success(f"Barcode : {barcode}")
                else:
                    st.error(f"Cannot read barcode")
                
                st.title("Product info")
                
                if info and info.get("status"):
                    st.write(info)
                else:
                    st.write("Product not found")

if __name__ == '__main__':
    main()
