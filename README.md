# Barcode-detection
This project aims to develop a deep learning model able to detect a barcode in a given image. The model behind it is Tiny YOLO 3, which a light version of YOLO 3 with less hidden layers in the neural network architecture. This helps significantly reduce the inference time, although its predictive accuracy is lower than YOLO 3 itself. For real time applications, this trade-off can be accepted in most cases.

## Description
Here are the 4 steps for this project :
- Implement Tiny YOLO 3 with pretrained weights (80 classes). Using transfer learning, train a model on a set of ~600 barcodes images (90% train / 10% validation).
- Use the model trained for inference on a new image.
- Use pyzbar (python library to read barcodes) to decode the barcode.
- Call OpenfoodFact API to retrieve informations about the product (for food products).
The final model can be tested in a streamlit app, by uploading an image and getting the resulting image with a bounding box over the barcode.

## Installation
1. Install python 3.8+.

2. Install zbar for Mac/Linux:

    Linux :

    ```
    sudo apt-get install libzbar0
    ```

    Mac (make sure brew is installed):

    ```
    brew install zbar
    ```

    The zbar DLLs are included with the Windows Python wheels.

3. Install the requirements
    ```
    pip install -r requirements.txt
    ```

## Inference
The inference result depends on some parameters tuning that can be made in settings.py file, especially for:
- score_threshold
- iou_threshold

These parameters can be changed before starting the app.
The app can be started like follows :

1. Launch streamlit app :
    ```
    streamlit run app.py
    ```
2. Upload image and click "Launch barcode detection"
3. If the barcode is detected, a bounding box will appear in the image around the barcode.
4. If the barcode is decoded, it will show in the screen.
5. If the OpenFoodFacts API contains information about the product, it will appear in the product info section.

## Training
