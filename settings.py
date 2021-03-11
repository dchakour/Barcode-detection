import numpy as np

class Settings:
    
    model = {
        "size":                 416, # image size - model w/ square images
        "anchors":              np.array([(10, 14), (23, 27), (37, 58),
                                        (81, 82), (135, 169), (344, 319)],
                                        np.float32) / 416,
        "masks":                np.array([[3, 4, 5], [0, 1, 2]]),
        "score_threshold":      0.5,
        "iou_threshold":        0.3,
        "max_boxes":            5,
        "weights":              "data/model/yolov3_train_final.tf",
    }
    
    train = {
        "batch_size":           32,
        "learning_rate":        1e-3,
        "epochs":               100,
        "weights_num_classes":  80, #Transfer learning with a different nb classes
        "train_dataset": "data/train.tf_record",
        "val_dataset": "data/validation.tf_record",
        "classes": "data/classes.csv",
        "pretrained_weights": "checkpoints/yolov3-tiny.tf",
        "final_weights": "data/model/yolov3_train_final.tf",
        "checkpoints": "checkpoints/yolov3_train_{epoch}.tf",
        "logs": "logs/",
        "run_eagerly": True # Set to True for debugging
    }
    
    class_names = ["Barcode"]