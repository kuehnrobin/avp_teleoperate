import time
from vuer import Vuer
from vuer.schemas import ImageBackground, Hands
from multiprocessing import Array, Process, shared_memory
import numpy as np
import asyncio
import cv2

from multiprocessing import context
Value = context._default_context.Value


class TeleVision:
    def __init__(self, binocular, img_shape, img_shm_name, wrist_img_shape=None, wrist_img_shm_name=None, cert_file="./cert.pem", key_file="./key.pem", ngrok=False):
        self.binocular = binocular
        self.img_height = img_shape[0]
        if binocular:
            self.img_width  = img_shape[1] // 2
        else:
            self.img_width  = img_shape[1]

        # Initialize wrist camera support
        self.wrist_cameras = wrist_img_shape is not None and wrist_img_shm_name is not None
        if self.wrist_cameras:
            self.wrist_img_height = wrist_img_shape[0]
            self.wrist_img_width = wrist_img_shape[1] // 2  # Assuming stereo wrist cameras
            existing_wrist_shm = shared_memory.SharedMemory(name=wrist_img_shm_name)
            self.wrist_img_array = np.ndarray(wrist_img_shape, dtype=np.uint8, buffer=existing_wrist_shm.buf)

        if ngrok:
            self.vuer = Vuer(host='0.0.0.0', queries=dict(grid=False), queue_len=3) #Change here for Quest
        else: 
            self.vuer = Vuer(host='0.0.0.0', cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3)

        self.vuer.add_handler("HAND_MOVE")(self.on_hand_move)
        self.vuer.add_handler("CAMERA_MOVE")(self.on_cam_move)

        existing_shm = shared_memory.SharedMemory(name=img_shm_name)
        self.img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=existing_shm.buf)
 
        if binocular:
            self.vuer.spawn(start=False)(self.main_image_binocular)
        else:
            self.vuer.spawn(start=False)(self.main_image_monocular)

        self.left_hand_shared = Array('d', 16, lock=True)
        self.right_hand_shared = Array('d', 16, lock=True)
        self.left_landmarks_shared = Array('d', 75, lock=True)
        self.right_landmarks_shared = Array('d', 75, lock=True)
        
        self.head_matrix_shared = Array('d', 16, lock=True)
        self.aspect_shared = Value('d', 1.0, lock=True)

        self.process = Process(target=self.vuer_run)
        self.process.daemon = True
        self.process.start()

    
    def vuer_run(self):
        self.vuer.run()

    async def on_cam_move(self, event, session, fps=60):
        try:
            self.head_matrix_shared[:] = event.value["camera"]["matrix"]
            self.aspect_shared.value = event.value['camera']['aspect']
        except:
            pass

    async def on_hand_move(self, event, session, fps=60):
        try:
            self.left_hand_shared[:] = event.value["leftHand"]
            self.right_hand_shared[:] = event.value["rightHand"]
            self.left_landmarks_shared[:] = np.array(event.value["leftLandmarks"]).flatten()
            self.right_landmarks_shared[:] = np.array(event.value["rightLandmarks"]).flatten()
        except: 
            pass
    
    async def main_image_binocular(self, session, fps=60):
        session.upsert @ Hands(fps=fps, stream=True, key="hands", showLeft=False, showRight=False)
        while True:
            display_image = cv2.cvtColor(self.img_array, cv2.COLOR_BGR2RGB)
            
            # Get left and right eye images
            left_eye_image = display_image[:, :self.img_width]
            right_eye_image = display_image[:, self.img_width:]
            
            # Add wrist camera overlays if available
            if self.wrist_cameras:
                wrist_display_image = cv2.cvtColor(self.wrist_img_array, cv2.COLOR_BGR2RGB)
                left_wrist_image = wrist_display_image[:, :self.wrist_img_width]
                right_wrist_image = wrist_display_image[:, self.wrist_img_width:]
                
                # Create overlays
                left_eye_image = self.create_overlay_image(left_eye_image, left_wrist_image, eye='left')
                right_eye_image = self.create_overlay_image(right_eye_image, right_wrist_image, eye='right')
            
            session.upsert(
                [
                    ImageBackground(
                        left_eye_image,
                        aspect=1.778,
                        height=1,
                        distanceToCamera=1,
                        # The underlying rendering engine supported a layer binary bitmask for both objects and the camera. 
                        # Below we set the two image planes, left and right, to layers=1 and layers=2. 
                        # Note that these two masks are associated with left eye’s camera and the right eye’s camera.
                        layers=1,
                        format="jpeg",
                        quality=50,
                        key="background-left",
                        interpolate=True,
                    ),
                    ImageBackground(
                        right_eye_image,
                        aspect=1.778,
                        height=1,
                        distanceToCamera=1,
                        layers=2,
                        format="jpeg",
                        quality=50,
                        key="background-right",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )
            # 'jpeg' encoding should give you about 30fps with a 16ms wait in-between.
            await asyncio.sleep(0.016 * 2)

    async def main_image_monocular(self, session, fps=60):
        session.upsert @ Hands(fps=fps, stream=True, key="hands", showLeft=False, showRight=False)
        while True:
            display_image = cv2.cvtColor(self.img_array, cv2.COLOR_BGR2RGB)
            
            # Add wrist camera overlays if available
            if self.wrist_cameras:
                wrist_display_image = cv2.cvtColor(self.wrist_img_array, cv2.COLOR_BGR2RGB)
                left_wrist_image = wrist_display_image[:, :self.wrist_img_width]
                right_wrist_image = wrist_display_image[:, self.wrist_img_width:]
                
                # For monocular, overlay both wrist cameras - left in left corner, right in right corner
                display_image = self.create_overlay_image(display_image, left_wrist_image, eye='left')
                display_image = self.create_overlay_image(display_image, right_wrist_image, eye='right')
            
            session.upsert(
                [
                    ImageBackground(
                        display_image,
                        aspect=1.778,
                        height=1,
                        distanceToCamera=1,
                        format="jpeg",
                        quality=50,
                        key="background-mono",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )
            await asyncio.sleep(0.016)

    @property
    def left_hand(self):
        return np.array(self.left_hand_shared[:]).reshape(4, 4, order="F")
        
    
    @property
    def right_hand(self):
        return np.array(self.right_hand_shared[:]).reshape(4, 4, order="F")
        
    
    @property
    def left_landmarks(self):
        return np.array(self.left_landmarks_shared[:]).reshape(25, 3)
    
    @property
    def right_landmarks(self):
        return np.array(self.right_landmarks_shared[:]).reshape(25, 3)

    @property
    def head_matrix(self):
        return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")

    @property
    def aspect(self):
        return float(self.aspect_shared.value)

    def create_overlay_image(self, main_image, wrist_image, eye='left'):
        """Create an overlay image with wrist camera in the bottom corner"""
        overlay = main_image.copy()
        
        if self.wrist_cameras and wrist_image is not None:
            # Get the actual width of the current image (could be full width for mono or half for stereo)
            current_img_width = main_image.shape[1]
            current_img_height = main_image.shape[0]
            
            # Resize wrist image to fit in corner (e.g., 1/4 of main image width)
            corner_width = current_img_width // 4
            corner_height = int(corner_width * self.wrist_img_height / self.wrist_img_width)
            
            # Ensure corner doesn't exceed main image height
            if corner_height > current_img_height // 3:
                corner_height = current_img_height // 3
                corner_width = int(corner_height * self.wrist_img_width / self.wrist_img_height)
            
            # Resize wrist image
            wrist_resized = cv2.resize(wrist_image, (corner_width, corner_height))
            
            # Calculate position for bottom corner
            if eye == 'left':
                # Left wrist in left bottom corner
                y_start = current_img_height - corner_height
                x_start = 0
            else:
                # Right wrist in right bottom corner
                y_start = current_img_height - corner_height
                x_start = current_img_width - corner_width
            
            # Add wrist image overlay
            overlay[y_start:y_start+corner_height, x_start:x_start+corner_width] = wrist_resized
        
        return overlay

if __name__ == '__main__':
    import os 
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    import threading
    from image_server.image_client import ImageClient

    # image
    img_shape = (480, 640 * 2, 3)
    img_shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
    img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=img_shm.buf)
    
    # Optional wrist images
    wrist_img_shape = (480, 640 * 2, 3)
    wrist_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(wrist_img_shape) * np.uint8().itemsize)
    wrist_img_array = np.ndarray(wrist_img_shape, dtype=np.uint8, buffer=wrist_img_shm.buf)
    
    img_client = ImageClient(tv_img_shape = img_shape, tv_img_shm_name = img_shm.name)
    image_receive_thread = threading.Thread(target=img_client.receive_process, daemon=True)
    image_receive_thread.start()

    # television
    tv = TeleVision(True, img_shape, img_shm.name, wrist_img_shape, wrist_img_shm.name)
    print("vuer unit test program running...")
    print("you can press ^C to interrupt program.")
    try:
        while True:
            time.sleep(0.03)
    finally:
        img_shm.unlink()
        img_shm.close()
        wrist_img_shm.unlink()
        wrist_img_shm.close()