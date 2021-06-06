import tensorflow as tf

def predict_from_image(model, image):
    return model.predict(image[tf.newaxis, ...])[0]