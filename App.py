from flask import Flask, render_template, Response
import cv2
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

camera = cv2.VideoCapture(0)  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)
classes = ["person","helmet","no_helmet"]
colors = np.random.uniform(0,255,size=(len(classes),3))
with tf.gfile.FastGFile('frozen_inference_graph.pb','rb') as f:
    graph_def=tf.GraphDef()
    graph_def.ParseFromString(f.read())
with tf.Session() as sess:
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')


def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, img = camera.read()  # read the camera frame
        if not success:
            break
        else:
            img=cv2.resize(img,(0,0),None,1,1)
            rows=img.shape[0]
            cols=img.shape[1]
            inp=cv2.resize(img,(220,220))
            print(inp)
            inp=inp[:,:,[2,1,0]]
            out=sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                sess.graph.get_tensor_by_name('detection_scores:0'),
                            sess.graph.get_tensor_by_name('detection_boxes:0'),
                            sess.graph.get_tensor_by_name('detection_classes:0')],
                            feed_dict={'image_tensor:0':inp.reshape(1, inp.shape[0], inp.shape[1],3)})
            print(out)
            num_detections=int(out[0][0])
            for i in range(num_detections):
                classId = int(out[3][0][i])
                score=float(out[1][0][i])
                bbox=[float(v) for v in out[2][0][i]]
                label=classes[classId]
                if (score>0.3):
                    x=bbox[1]*cols
                    y=bbox[0]*rows
                    right=bbox[3]*cols
                    bottom=bbox[2]*rows
                    color=colors[classId]
                    cv2.rectangle(img, (int(x), int(y)), (int(right),int(bottom)), color, thickness=2)
                    cv2.rectangle(img, (int(x), int(y)), (int(right),int(y+30)),color, -1)
                    cv2.putText(img, str(label),(int(x), int(y+25)),1,2,(255,255,255),2)
                    frame = cv2.imencode('jpg',img)[1].tobytes()
                    yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' +frame+ b'\r\n')  # read the camera frame
        # concat frame one by one and show result
        


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)