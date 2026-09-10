from pathlib import Path
import json, sys
import numpy as np
import tensorflow as tf

ROOT=Path(__file__).resolve().parents[1]
model=tf.keras.models.load_model(ROOT/'models/quality_model.keras')
classes=json.loads((ROOT/'models/class_names.json').read_text())
path=Path(sys.argv[1])
img=tf.keras.utils.load_img(path, target_size=(128,128))
x=tf.keras.utils.img_to_array(img)[None,...]
probs=model.predict(x, verbose=0)[0]
i=int(np.argmax(probs))
print({"class": classes[i], "confidence": round(float(probs[i]),4)})
