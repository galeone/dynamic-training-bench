import unittest
import tensorflow as tf

from dytb.models.predefined.VGG import VGG
from dytb.inputs.images import read_image


class TestFeatureExtractors(unittest.TestCase):

    def test_classifier(self):
        model = VGG()
        image = tf.image.resize_bilinear(
            tf.expand_dims(
                read_image("images/nocat.png", channel=3, image_type="png"),
                axis=0), (32, 32))
        features = model.evaluator.extract_features(
            checkpoint_path="../log/VGG/CIFAR-10_Momentum/best/",
            inputs=image,
            layer_name="VGG/pool1/MaxPool:0",
            num_classes=10)
        self.assertEqual(features.shape, (1, 16, 16, 64))


if __name__ == '__main__':
    unittest.main()
