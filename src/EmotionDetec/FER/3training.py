module1 = "2CNN_Model"
module2 = "1preprocess"
mod1 = __import__(module1) 
mod2 = __import__(module2)

from mod1 import *
from mod2 import *
import pandas as pd 
# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=2)

# Save the trained model
model.save('emotion_detection_model.h5')
