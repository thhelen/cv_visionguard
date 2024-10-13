import cv2
from threading import Thread

class MyVideoCapture:
    def __init__(self, video_source=0,width=None,height=None):
        # Open the video source
        self.vid  = None
        self.video_src = video_source
        self.stopped = False
        self.width=width
        self.height = height

    def read(self):
        return self.ret,self.frame
    
    def get_frame(self, out_p):
        self.vid = cv2.VideoCapture(self.video_src)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", self.video_src)

        try:
            while True:
                
                if  self.vid.isOpened():
                    self.ret, self.frame = self.vid.read()
                    if self.ret:
                        # Return a boolean success flag and the current frame converted to BGR
                        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                        self.frame = cv2.resize(self.frame,(self.width,self.height))
                        out_p.send((True,self.frame))
                    else:
                        out_p.send((False,None))
                        break
        except AttributeError as e:
            print("Vid is closed !!! Shutting down process.")
    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
