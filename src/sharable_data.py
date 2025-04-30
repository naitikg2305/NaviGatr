# queues.py Use across NaviGatr modules.
# All grabbed frames will be captured and duplicated for all components to operate on each
# frame in parallel.
#
# Example: 1) Live Video feed is being captured
#          2) Feed is sliced and frame is grabbed from slice
#          3) Frame is duplicated with a shallow copy and a copy is placed in every component's frame queue
#          4) Every component grabs the frame from its queue and runs ML model on it
#          5) Resulting data is place on the respective component's result queue
#          6) Frame on 'frame_queue' is popped indicating frame has been handled
#          7) All component's result queues are popped and data merging is handled
#          *) Resulting merged data gets pushed on 'result_queue'
from collections import deque
import threading

thread_lock = threading.Lock()

class RecentFrameQueue:
    def __init__(self, max_size=1):
        """
        __init__ constructor for RecentFrameQueue

        :param max_size: maximum number of frames to keep in the queue, defaults to 1
        """
        self.max_size = max_size
        # Validate max_size
        if max_size < 1 or not isinstance(max_size, int):
            raise ValueError("max_size must be at least 1 and an integer")
        self.queue = deque(maxlen=max_size)

    def __len__(self):
        return len(self.queue)

    def put(self, frame):
        self.queue.append(frame)

    def get(self):
        res = None
        try:
            res = self.queue.popleft()
        except:
            res = self.queue.pop()
            print(f"SharableData: Successfully popped frame from right...")
            pass
        return res
    

frame_queue = RecentFrameQueue()  # Captured frame
obj_queue = RecentFrameQueue(max_size=1)
depth_queue = RecentFrameQueue(max_size=1)
emot_queue = RecentFrameQueue(max_size=1)

obj_res_queue = RecentFrameQueue(max_size=1)
depth_res_queue = RecentFrameQueue(max_size=1)
emot_res_queue = RecentFrameQueue(max_size=1)

# Result queues (i.e. post-inferencing) used for generating saved videos
# obj_res_queue = deque()
# depth_res_queue = deque()
# emot_res_queue = deque()
