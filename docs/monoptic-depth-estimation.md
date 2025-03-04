# MiDaS
- Trained on multiple data sets including 3D films
- Trained for use with images not video
- Robust to diverse datasets
- Models freely available
- It seems that according to their paper, their model outperforms all other proposed algorithms
- Has various models with their speed vs. accuracy tradeoffs
- However, only provides relative depth estimates, which means we will need to manually calculate metric, perhaps via a reference object?
- **ZoeVision** is built on top of MiDaS and provides metric depth estimation
- **Sources:**
	- https://arxiv.org/pdf/1907.01341v3
	- https://www.youtube.com/watch?v=D46FzVyL9I8
	- https://github.com/isl-org/MiDaS
	- https://github.com/isl-org/ZoeDepth
# MonoDepth2
- Has metric depth perception available with mono+stereo models
- Also, has depth estimation models
- ManyDepth is a follow up that utilizes 2 adjacent frames to predict depth maps
- Sources:
	- https://github.com/nianticlabs/monodepth2
	- https://adityang5.medium.com/monocular-depth-estimation-in-python-using-monodepth2-and-manydepth-75170dfd4bb2

# DPT
- Seems to be the newest model, though it is no longer under management
- Provides relative depth maps with better accuracy than even MiDaS
- Sources:
	- https://arxiv.org/pdf/2103.13413

# Conclusion
It seems reasonable to start with ZoeDepth as it provides metric depth estimation and it's built on MiDaS, which seems to be the best maintained.