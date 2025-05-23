import numpy as np
import os
import time
import threading
import logging
import json
from datetime import datetime
import h5py

class PoseLogger:
    """
    A background logger for robot pose data that saves information in HDF5 format
    for later analysis with tools like matplotlib.
    """
    
    def __init__(self, log_dir='./logs', log_interval=0.1, max_buffer_size=1000):
        """
        Initialize the pose logger
        
        Args:
            log_dir (str): Directory to store logs
            log_interval (float): How often to write to disk (seconds)
            max_buffer_size (int): Maximum size of the buffer before writing to disk
        """
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.max_buffer_size = max_buffer_size
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"pose_data_{timestamp}.h5")
        
        # Initialize metrics logging
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[logging.FileHandler(os.path.join(log_dir, f"pose_log_{timestamp}.txt"))]
        )
        self.logger = logging.getLogger('pose_logger')
        self.logger.info(f"Pose logger initialized. Data will be saved to {self.log_file}")
        
        # Data buffers
        self.buffer = {
            'timestamp': [],
            'frame_number': [],
            'head_rmat': [],
            'left_wrist': [],
            'right_wrist': [],
            'left_hand_shape': [],
            'right_hand_shape': [],
            'left_wrist_height': [],
            'right_wrist_height': []
        }
        
        # Initialize the HDF5 file
        with h5py.File(self.log_file, 'w') as f:
            # Create dataset groups
            f.create_group('head_rmat')
            f.create_group('left_wrist')
            f.create_group('right_wrist')
            f.create_group('left_hand_shape')
            f.create_group('right_hand_shape')
            f.create_group('metadata')
            
            # Create metadata
            f['metadata'].attrs['created'] = timestamp
            f['metadata'].attrs['description'] = 'Robot pose data log'
        
        self.frame_number = 0
        self.running = False
        self.lock = threading.Lock()
    
    def start(self):
        """Start the background logging thread"""
        self.running = True
        self.logger.info("Starting background pose logging")
        self.bg_thread = threading.Thread(target=self._background_save, daemon=True)
        self.bg_thread.start()
        return self
    
    def stop(self):
        """Stop the background logging thread and flush all data"""
        self.running = False
        if hasattr(self, 'bg_thread'):
            self.bg_thread.join(timeout=5.0)
        self._save_buffer()
        self.logger.info("Pose logging stopped and data flushed to disk")
    
    def log_pose(self, head_rmat, left_wrist, right_wrist, left_hand, right_hand):
        """
        Log a set of pose matrices
        
        Args:
            head_rmat: Head rotation matrix
            left_wrist: Left wrist pose matrix
            right_wrist: Right wrist pose matrix
            left_hand: Left hand data
            right_hand: Right hand data
        """
        if not self.running:
            return
        
        with self.lock:
            self.buffer['timestamp'].append(time.time())
            self.buffer['frame_number'].append(self.frame_number)
            self.buffer['head_rmat'].append(head_rmat.flatten())
            self.buffer['left_wrist'].append(left_wrist.flatten())
            self.buffer['right_wrist'].append(right_wrist.flatten())
            self.buffer['left_hand_shape'].append(np.array(left_hand.shape))
            self.buffer['right_hand_shape'].append(np.array(right_hand.shape))
            
            # Extract heights for quick access in analysis
            self.buffer['left_wrist_height'].append(left_wrist[2, 3])
            self.buffer['right_wrist_height'].append(right_wrist[2, 3])
            
            # Log warnings if heights exceed threshold
            if left_wrist[2, 3] > 0.2:
                self.logger.warning(f"Frame {self.frame_number}: Left wrist height ({left_wrist[2, 3]:.3f}) above threshold")
            if right_wrist[2, 3] > 0.2:
                self.logger.warning(f"Frame {self.frame_number}: Right wrist height ({right_wrist[2, 3]:.3f}) above threshold")
            
            self.frame_number += 1
        
        # Check buffer size and save if needed
        if len(self.buffer['timestamp']) >= self.max_buffer_size:
            self._save_buffer()
    
    def _background_save(self):
        """Background thread that periodically saves data to disk"""
        while self.running:
            time.sleep(self.log_interval)
            if len(self.buffer['timestamp']) > 0:
                self._save_buffer()
    
    def _save_buffer(self):
        """Save the current buffer to disk"""
        with self.lock:
            if len(self.buffer['timestamp']) == 0:
                return
            
            # Make local copy of buffer
            local_buffer = {k: np.array(v) for k, v in self.buffer.items()}
            
            # Clear buffer
            for k in self.buffer:
                self.buffer[k] = []
        
        # Save to HDF5 file
        try:
            with h5py.File(self.log_file, 'a') as f:
                # Determine the next index to write
                start_idx = f['metadata'].attrs.get('next_index', 0)
                end_idx = start_idx + len(local_buffer['timestamp'])
                
                # Create or resize datasets as needed
                for key in ['head_rmat', 'left_wrist', 'right_wrist']:
                    data_shape = local_buffer[key][0].shape
                    dataset_name = f'{key}/data'
                    
                    if dataset_name in f:
                        dataset = f[dataset_name]
                        if end_idx > dataset.shape[0]:
                            dataset.resize((end_idx + 1000,) + data_shape)
                    else:
                        dataset = f.create_dataset(
                            dataset_name, 
                            shape=(end_idx + 1000,) + data_shape,
                            maxshape=(None,) + data_shape,
                            dtype=np.float32,
                            chunks=True,
                            compression="gzip"
                        )
                    
                    # Write data
                    for i, data in enumerate(local_buffer[key]):
                        dataset[start_idx + i] = data
                
                # Save timestamps and frame numbers
                for key in ['timestamp', 'frame_number', 'left_wrist_height', 'right_wrist_height']:
                    dataset_name = f'metadata/{key}'
                    if dataset_name in f:
                        dataset = f[dataset_name]
                        if end_idx > dataset.shape[0]:
                            dataset.resize((end_idx + 1000,))
                    else:
                        dataset = f.create_dataset(
                            dataset_name, 
                            shape=(end_idx + 1000,),
                            maxshape=(None,),
                            dtype=np.float64 if key == 'timestamp' else np.int32,
                            chunks=True
                        )
                    
                    # Write data
                    dataset[start_idx:end_idx] = local_buffer[key]
                
                # Update metadata
                f['metadata'].attrs['next_index'] = end_idx
                f['metadata'].attrs['last_update'] = datetime.now().strftime("%Y%m%d_%H%M%S")
                
        except Exception as e:
            self.logger.error(f"Error saving pose data: {e}")

def analyze_pose_logs(log_file):
    """
    Helper function to analyze saved pose logs
    
    Args:
        log_file (str): Path to the HDF5 log file
    """
    import matplotlib.pyplot as plt
    
    with h5py.File(log_file, 'r') as f:
        timestamps = f['metadata/timestamp'][:]
        frame_numbers = f['metadata/frame_number'][:]
        left_heights = f['metadata/left_wrist_height'][:]
        right_heights = f['metadata/right_wrist_height'][:]
        
        # Convert timestamps to relative time in seconds
        rel_time = timestamps - timestamps[0]
        
        # Plot heights
        plt.figure(figsize=(12, 6))
        plt.plot(rel_time, left_heights, label='Left Wrist Height')
        plt.plot(rel_time, right_heights, label='Right Wrist Height')
        plt.axhline(y=0.2, color='r', linestyle='--', label='Threshold')
        plt.xlabel('Time (s)')
        plt.ylabel('Height (m)')
        plt.title('Wrist Heights Over Time')
        plt.legend()
        plt.grid(True)
        
        output_dir = os.path.dirname(log_file)
        plt.savefig(os.path.join(output_dir, 'wrist_heights.png'))
        plt.show()