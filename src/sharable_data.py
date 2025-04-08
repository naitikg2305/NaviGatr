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
import queue

frame_queue = queue.LifoQueue(maxsize=5)  # Captured frame
result_queue = queue.Queue(maxsize=5) # Resulting frame data

# Below are the queues for depth detection


# Below are the queues for object detection