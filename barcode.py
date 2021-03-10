import requests, json
import re
from pyzbar.pyzbar import decode
from PIL import Image

def get_barcode(image):
    """
    Read the barcode from the picture and return the barcode number in EAN 13 format
    """
    barcode = info = None
    img_raw = Image.open(image)
    decoded = decode(img_raw)
    
    if decoded:
        try:
            barcode = int(re.findall("[+-]?[0-9][0-9]*|0$",
                                    str(decoded[0].data))[0])
            info = _get_product_info(barcode)
        except:
            pass

    return barcode, info

def _get_product_info(barcode):
    """
    Query a product by the barcode number the openFoodFact DB trought API 
    """
    address = "https://world.openfoodfacts.org/api/v0/product/{}.json".format(barcode)
    #print(address) use for logging
    r = requests.get(address)
    return json.loads(r.text)
