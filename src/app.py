#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import sys
import os
import json
import argparse
import urllib.request
import uuid
import numpy
import traceback
from PIL import Image
from bottle import route, run, request, response, hook, HTTPResponse
from envparse import env
from lib.Predictor import Predictor
from lib.MyEncoder import MyEncoder
# from lib.IO import IOManager

from io import BytesIO
import base64

import paho.mqtt.client as mqtt

# model_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trained"))

env.read_envfile()

# WEIGHT_FILE = env("WEIGHT_FILE", os.path.join(model_dir, "mask_rcnn_msg.h5"))
# LABEL_FILE = env.str("LABEL_FILE", default=os.path.join(model_dir, "names.txt"))

host = '192.168.179.7'
port = 1883
topic_image = '/pub/gun/image'
topic_predicted = '/sub/gun/predicted'

model_path = 'model_data/yolo-tiny.h5'
anchors_path = 'model_data/tiny_yolo_anchors.txt'
classes_path = 'model_data/coco_classes.txt'

PORT = env.int("PORT", default=8080)

hasInternalError = False

predictor = Predictor()
predictor.load_model(model_path, anchors_path, classes_path)

def on_connect(client, userdata, flags, respons_code):
    print('status {0}'.format(respons_code))
    client.subscribe(topic_image)

def on_message(client, userdata, msg):
    print(msg.topic + ' ' + str(msg.payload))
    # image_base64 = str(msg.payload)
    image_base64 = msg.payload
    try:
        image = Image.open(BytesIO(base64.b64decode(image_base64)))
        # image = Image.open(cStringIO.StringIO(image_base64))
        if image is None:
            raise Exception('image is invalid')
        results = run_detect(image)
        client.publish(topic_predicted, results)
    except Exception as e:
        print('load model error', e)

        t, v, tb = sys.exc_info()
        print(traceback.format_exception(t,v,tb))
        print(traceback.format_tb(e.__traceback__))

def run_detect(image):
    items = []
    try:
        results = predictor.predict(image)
        for result in results: # TODO: 要確認
            for box in result:
                items.append({
                    "id": box.class_index,
                    "name": box.class_label,
                    "detected_area": [
                        box.left,
                        box.top,
                        box.width,
                        box.height
                    ]
                })
    except Exception as ex:
        print(ex)
        print(traceback.format_exc())
        return  HTTPResponse(status=500)
    
    # build responce
    result_content = {
        "items": items
    }
    # {
    #     "items": [
    #         {
    #             "id": 1,
    #             "name": "heavy_010_BatteringRam",
    #             "detected_area": [ 10, 5, 200, 250 ]
    #         }
    #     ]
    # }
    print(json.dumps(result_content, indent=2, cls=MyEncoder)) # debugprint
    return json.dumps(
        result_content,
        sort_keys=True,
        ensure_ascii=False,
        indent=2,
        cls=MyEncoder
    )  # FIXME: ensure_ascii

if __name__ == '__main__':
    # Publisherと同様に v3.1.1を利用
    client = mqtt.Client(protocol=mqtt.MQTTv311)

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(host, port=port, keepalive=60)

    # 待ち受け状態にする
    client.loop_forever()

# -----------

# @hook('before_request')
# def before_request():
#     # IP制限
#     if not IPAC:
#         return
#     remote_addr = request.environ.get('REMOTE_ADDR')
#     if remote_addr not in ipac_list:
#         print(request.environ['PATH_INFO'], " -> /error_403")
#         request.environ['PATH_INFO'] = '/error_403'

@hook('after_request')
def after_request():
    # CORS settings
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'


# -----------
# request hundler

@route('/error_403', method=['GET', 'POST'])
def error_403():
    print("request was rejected.", request.environ.get('REMOTE_ADDR'))
    return HTTPResponse(status=403)

@route('/api/v1/health', method='GET')
def api_health_check():
    r = None
    if not hasInternalError:
        r = HTTPResponse(status=200)
        # TODO: yoloモデルのテスト実行
    else:
        r = HTTPResponse(status=500)
    return r


@route('/api/v1/detector', method='POST')
def api_predict():
    """REST API : predict
    """
    response.content_type = 'application/json'

    # 画像を取得
    upload = request.files.get('image')
    if upload == None:
        print('image filed is not found.')
        return  HTTPResponse(status=400)
    if not upload.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        print('File extension not allowed!', upload.filename.lower())
        return  HTTPResponse(status=400)
    
    # tmpdir = "./tmp"
    # upload.file     # Open file(-like) object (BytesIO buffer or temporary file)
    # upload.filename # Name of the file on the client file system
    # image = load_img(upload.file, target_size=(416, 416))
    image = Image.open(upload.file)

    results = run_detect(image)

    return results

# -----------
# bootstrap

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bottle Server')
    parser.add_argument('--port', dest='port', type=int, default=PORT, help='port number (default: 58080')
    args = parser.parse_args()
    run(host='0.0.0.0', port=args.port, debug=True)
