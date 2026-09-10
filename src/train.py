from pathlib import Path
import json
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "images"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
IMG_SIZE=(128,128)
BATCH=16
train_ds = tf.keras.utils.image_dataset_from_directory(DATA, validation_split=0.2, subset="training", seed=42, image_size=IMG_SIZE, batch_size=BATCH)
val_ds = tf.keras.utils.image_dataset_from_directory(DATA, validation_split=0.2, subset="validation", seed=42, image_size=IMG_SIZE, batch_size=BATCH)
class_names = train_ds.class_names
aug = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.05), tf.keras.layers.RandomZoom(0.1)])
base = tf.keras.applications.MobileNetV2(input_shape=(128,128,3), include_top=False, weights=None)
base.trainable = True
inputs=tf.keras.Input(shape=(128,128,3))
x=aug(inputs)
x=tf.keras.applications.mobilenet_v2.preprocess_input(x)
x=base(x, training=True)
x=tf.keras.layers.GlobalAveragePooling2D()(x)
x=tf.keras.layers.Dropout(0.2)(x)
outputs=tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
model=tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
h=model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks, verbose=2)
model.save(MODELS / "quality_model.keras")
(MODELS / "class_names.json").write_text(json.dumps(class_names))
print({"classes": class_names, "best_val_accuracy": max(h.history["val_accuracy"])})
