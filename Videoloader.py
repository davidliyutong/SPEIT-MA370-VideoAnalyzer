import cv2
from utils import BGR_2_gray


class VideoLoader:
    def __init__(self, video_path=None):
        self.video_path = video_path
        if self.video_path is not None:
            self.camera = cv2.VideoCapture(video_path)
            self._update_info()

    def open(self, video_path):
        if self.camera is not None:
            self.camera.release()
        self.video_path = video_path
        self.camera = cv2.VideoCapture(video_path)
        self._update_info()

    def _update_info(self):
        self.fps = self.camera.get(cv2.CAP_PROP_FPS)
        self.size = (int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def get_size(self):
        self._update_info()
        return self.size

    def get_fps(self):
        self._update_info()
        return self.fps

    def read(self):
        if self.camera is None:
            return None
        res, frame = self.camera.read()
        if not res:
            print('END_OF_VIDEO')
            self.camera.release()
            self.camera = None
            return None
        return frame

    def read_gray(self):
        frame = self.read()
        if frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
