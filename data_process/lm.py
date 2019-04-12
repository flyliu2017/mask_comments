import json
import os
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.contrib import predictor
os.environ['CUDA_VISIBLE_DEVICES']='0'
app = Flask(__name__)

print("# Load lm model...")
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=config)
predict_fn = predictor.from_saved_model("/data/xueyou/car/comment/lm/score/0/")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    ret = predict_fn(data)['ppl']
    
    ret= [float(v) for v in ret]
    return json.dumps(ret)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9193, debug=False,threaded=True)