import tensorflow as tf
import numpy as np
import cv2
import json

# Monkey-patch untuk mengatasi error 'quantization_config' pada Keras 3
original_dense_from_config = tf.keras.layers.Dense.from_config

def patched_dense_from_config(cls, config):
    if "quantization_config" in config:
        del config["quantization_config"]
    return original_dense_from_config(config)

tf.keras.layers.Dense.from_config = classmethod(patched_dense_from_config)

# Custom Layer yang dibutuhkan untuk me-load model Keras
@tf.keras.utils.register_keras_serializable()
class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name='ctc_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name='ctc_loss')

    def call(self, y_true, y_pred, label_length, training=None):
        batch_size = tf.shape(y_pred)[0]
        timesteps = tf.shape(y_pred)[1]
        input_length = tf.ones(shape=(batch_size,), dtype=tf.int32) * tf.cast(timesteps, tf.int32)

        loss = tf.nn.ctc_loss(
            labels=tf.cast(y_true, tf.int32),
            logits=y_pred,
            label_length=tf.cast(label_length, tf.int32),
            logit_length=input_length,
            logits_time_major=False,
            blank_index=0
        )

        if training:
            self.loss_tracker.update_state(loss)

        return tf.reduce_mean(loss)
        
    def get_config(self):
        return super().get_config()

class CRNNInference:
    def __init__(self, weight_path: str, vocab_path: str):
        # Load vocab
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            # JSON keys bertipe string ("0", "1", dst), ubah jadi integer untuk dicocokkan dengan output model
            self.num_to_char = {int(k): v for k, v in vocab_data["num_to_char"].items()}
            
        # Load model CRNN
        self.model = tf.keras.models.load_model(
            weight_path,
            custom_objects={'CTCLayer': CTCLayer}
        )
        
        # Ukuran input yang diharapkan model berdasarkan notebook training
        self.target_h = 64
        self.target_w = 480

    def _preprocess_image(self, img_array):
        # Convert BGR (OpenCV) ke RGB
        img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        # Resize dengan aspect ratio
        h, w = img.shape[:2]
        ratio = min(self.target_w / w, self.target_h / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        img = cv2.resize(img, (new_w, new_h))

        # Padding (putih)
        padded = np.ones((self.target_h, self.target_w, 3), dtype=np.uint8) * 255
        y_offset = (self.target_h - new_h) // 2
        x_offset = 0
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img

        # Normalisasi ke [0, 1]
        padded = padded.astype(np.float32) / 255.0
        
        # Tambah dimensi batch: (1, 64, 480, 3)
        return np.expand_dims(padded, axis=0)

    def _ctc_decode(self, y_pred):
        # TF Greedy Decoder
        input_len = np.ones(y_pred.shape[0]) * y_pred.shape[1]
        decoded, _ = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(y_pred, perm=[1, 0, 2]), 
            sequence_length=input_len.astype(np.int32)
        )
        dense = tf.sparse.to_dense(decoded[0], default_value=-1)
        
        results = []
        for seq in dense.numpy():
            text = ''
            for idx in seq:
                if idx == -1:
                    break
                if idx in self.num_to_char and idx != 0:  # skip blank
                    text += self.num_to_char[idx]
            results.append(text)
        return results

    def predict_text(self, cropped_img_array):
        """Menerima cropped image (numpy array BGR dari YOLO), mengembalikan raw text."""
        preprocessed = self._preprocess_image(cropped_img_array)
        y_pred = self.model.predict(preprocessed, verbose=0)
        decoded_texts = self._ctc_decode(y_pred)
        return decoded_texts[0] if len(decoded_texts) > 0 else ""
