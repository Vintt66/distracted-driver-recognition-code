"""Main settings for the distracted-driver experiments."""


# Class order matches the model output indexes.
CLASS_NAMES = [
    "Safe Driving",
    "Text Right",
    "Phone Right",
    "Text Left",
    "Phone Left",
    "Adjusting Radio",
    "Drinking",
    "Reaching Behind",
    "Hair or Makeup",
    "Talking to Passenger",
]

NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 224

INTERNAL_VALIDATION_FRACTION = 0.20
INTERNAL_SPLIT_SEED = 42
EXPECTED_OFFICIAL_TRAIN_IMAGES = 12_555
EXPECTED_INTERNAL_TRAIN_IMAGES = 10_044
EXPECTED_INTERNAL_VALIDATION_IMAGES = 2_511

SEEDS = [42, 43, 44]

BATCH_SIZE = 32
MAX_EPOCHS = 30
WEIGHT_DECAY = 0.0001
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.001

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


MODEL_SETTINGS = {
    "custom_cnn": {
        "learning_rate": 0.001,
        "pretrained": False,
        "expected_parameters": 391_466,
    },
    "resnet18": {
        "learning_rate": 0.0001,
        "pretrained": True,
        "expected_parameters": 11_181_642,
    },
    "mobilenet_v3_small": {
        "learning_rate": 0.0001,
        "pretrained": True,
        "expected_parameters": 1_528_106,
    },
}
