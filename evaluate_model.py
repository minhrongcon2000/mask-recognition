from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

FILENAME = "nose_model/resnet50_full_face/model-best.h5"
print(FILENAME)
model = load_model(FILENAME)
test_data_gen = ImageDataGenerator(
    rescale=1.0/255.
)

test_data_generator = test_data_gen.flow_from_directory(
    "dataset/sample",
    batch_size=32,
    target_size=(224, 224)
)

model.evaluate(test_data_generator)
