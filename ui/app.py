import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import numpy as np
from videoCapture import MyVideoCapture
from visionGuardProcessor import VisionGuardProcessor
from multiprocessing import Process, Queue, Pipe
import argparse

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Vision Guard.')
parser.add_argument('--vid', default="",
                    help='path of the video')



class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("455x720")
        self.width = 405
        self.height = 720
        # self.window.geometry("720x405")
        # self.width = 720
        # self.height = 455
        self.video_source = video_source
        self.frame_count =0
        # open video source (by default this will try to open the computer webcam)
        # self.vid_q = Queue()
        # self.vis_q = Queue()
        self.vid_p_send , self.vid_p_recv = Pipe()
        self.vis_p_send, self.vis_p_recv = Pipe()
       
        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window,width=self.width,height = self.height)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)


        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 1000//15

        self.vid = MyVideoCapture(self.video_source,self.width,self.height)

        self.visionguard =VisionGuardProcessor()

        self.vid_processor = Process(target=self.vid.get_frame,args=[self.vid_p_send])
        self.visionguard_processor = Process(target=self.visionguard.process,args=[self.vid_p_recv,self.vis_p_send] )
        self.vid_processor.deamon = True
        self.visionguard_processor.deamon = True
        self.vid_processor.start()
        self.visionguard_processor.start()

        self.update()

        self.window.protocol('WM_DELETE_WINDOW', self.destroy) 
        self.window.mainloop()

    def destroy(self):
        print("Shutting down threads")
        self.visionguard_processor.terminate()
        self.vid_processor.terminate()
        self.vid_processor.join()
        self.visionguard_processor.join()
        self.vid_processor.close()
        self.visionguard_processor.close()
        self.window.destroy()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.read()

        
        cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


    def update(self):
        # Get a frame from the video source
        # ret,frame = self.vid.read()
        # if not self.vis_q.empty():
        ret ,frame = self.vis_p_recv.recv()
        if not ret:
            self.destroy()
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
        self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
        
        self.window.after(self.delay, self.update)





if __name__ == "__main__":
    # Create a window and pass it to the Application object
    args = parser.parse_args()
    
    # video_path = "/Users/shunya/Project/visionguard/ui/videos/IMG_0347.MOV"

    App(tkinter.Tk(), "VisionGuard-beta-v0.009",args.vid)