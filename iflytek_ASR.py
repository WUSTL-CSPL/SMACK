import websocket
import hashlib
import base64
import hmac
import json
import _thread as thread
import re
import time
import ssl
import sys
import warnings
from datetime import datetime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
from time import mktime

warnings.filterwarnings("ignore")

STATUS_FIRST_FRAME = 0  # The identity of the first frame
STATUS_CONTINUE_FRAME = 1  # Intermediate frame identification
STATUS_LAST_FRAME = 2  # The identity of the last frame


class Ws_Param(object):
    # Initializing
    def __init__(self, APPID, APIKey, APISecret, AudioFile):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.AudioFile = AudioFile

        self.CommonArgs = {"app_id": self.APPID}
        
        # "language": Languages can be added trial or purchase in the console
        self.BusinessArgs = {"domain": "iat", "language": "en_us", "accent": "mandarin", "vinfo":1,"vad_eos":10000}
        

    def create_url(self):
        url = 'wss://iat-api-sg.xf-yun.com/v2/iat'
        # Generating a timestamp in RFC1123 format
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # Combining Strings
        signature_origin = "host: " + "iat-api-sg.xf-yun.com" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/iat " + "HTTP/1.1"
        # Hmac-sha256 is used for encryption
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        # Combines the request of authentication parameters into a dictionary
        v = {
            "authorization": authorization,
            "date": date,
            "host": "iat-api-sg.xf-yun.com"
        }
        # Concatenate the authentication parameters and generate the URL
        url = url + '?' + urlencode(v)
        
        # print(f"Constructed URL: {url}")  # Debug use only

        return url


# the websocket message has been received and handling
def on_message(ws, message):
    
    global message_received
    message_received = True
    
    try:
        code = json.loads(message)["code"]
        sid = json.loads(message)["sid"]
        if code != 0:
            errMsg = json.loads(message)["message"]
            print("sid:%s call error:%s code is:%s" % (sid, errMsg, code))

        else:
            data = json.loads(message)["data"]["result"]["ws"]
            global result
            # result = ''
            for i in data:
                for w in i["cw"]:
                    result += w["w"]
                    
    except Exception as e:
        print("receive message, but parse exception:", e)


# the error websocket message has been received and handling
def on_error(ws, error):
    print("### error:", error)
    return;


# the closed websocket message has been received and handling
def on_close(ws, *args):
    # print("### closed ###")  # Debug use only
    return;


# The connecting websocket message has been received and handling
def on_open(ws, wsParam):
    
    # print("WebSocket connection opened")  # Debug use only
    
    def run(*args):
        frameSize = 8000  # The audio size of each frame
        intervel = 0.04  # Send audio intervals (unit: S)
        status = STATUS_FIRST_FRAME  # The status information of the audio, identifying whether the audio is the first one, Intermediate one, or last frame

        with open(wsParam.AudioFile, "rb") as fp:
            while True:
                buf = fp.read(frameSize)
                if not buf:
                    status = STATUS_LAST_FRAME
                #  the first frame
                if status == STATUS_FIRST_FRAME:
                    # The first frame must be sent
                    d = {"common": wsParam.CommonArgs,
                         "business": wsParam.BusinessArgs,
                         "data": {"status": 0, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                    d = json.dumps(d)
                    # print(f"Sending message to server: {json.dumps(d)}")  # Debug use only
                    ws.send(d)
                    status = STATUS_CONTINUE_FRAME
                # Intermediate frame
                elif status == STATUS_CONTINUE_FRAME:
                    d = {"data": {"status": 1, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                    # print(f"Sending message to server: {json.dumps(d)}")  # Debug use only
                    ws.send(json.dumps(d))
                # the last frame
                elif status == STATUS_LAST_FRAME:
                    d = {"data": {"status": 2, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                    # print(f"Sending message to server: {json.dumps(d)}")  # Debug use only
                    ws.send(json.dumps(d))
                    time.sleep(1)
                    break
                # Analog audio sampling delay
                time.sleep(intervel)
        ws.close()

    thread.start_new_thread(run, ())


def iflytek_ASR(audio_file):
    
    # Your APPID/APIKey/APISecret
    wsParam = Ws_Param(APPID='ga0d4415', APIKey='607ae20f1957cfcad59efd4ab3282bd1',
                APISecret='7b11808df6792be7f3b7015ee7598679',
                AudioFile=audio_file)
    
    max_retries = 5  # Maximum number of retries
    retry_count = 0  # Current retry count
    
    global result
    global message_received
    result = ''

    while retry_count < max_retries:
        try:
            
            # Reset the flag for each attempt
            message_received = False

            # Access from the Console to find ASR webapi
            websocket.enableTrace(False)
            wsUrl = wsParam.create_url()
            ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
            ws.on_open = lambda ws: on_open(ws, wsParam)
            ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

            result = re.sub(r'[^\w\s]', '', result)
            result = result.upper()

            # If message was received, break the loop regardless of result content
            if message_received:
                break
                
        except Exception as e:
            print(f"Error in attempt {retry_count + 1}: {e}")
        
        retry_count += 1
        time.sleep(5)  # Sleep a bit before retrying
    
    # If after all retries result is still empty
    if result == '':
        result = 'NA'
    
    print(f'iflytek ASR Result after {retry_count} connection retries: {result}')
    return result

# For testing purposes
if __name__ == "__main__":
    
    audio_file = sys.argv[1]
    
    result = iflytek_ASR(audio_file)
