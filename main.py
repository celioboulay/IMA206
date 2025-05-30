import numpy as np
import matplotlib.pyplot as plt
import dec.dec as dec
import pandas as pd


####### On recupere les features de jsp ou

img_dir = 'Data'

model = NotImplemented
features = model(img_dir)


####### On initialize dec

pre_trained_SAE = NotImplemented

init = pre_trained_SAE(features)


######## on run dec jusqua converence

TMM = dec.TMM()

TMM.forward()
### en gros la boucle



final_features = NotImplemented # TMM.call ou jsp quoi pour avoir le dernier etat des features

######### plot des resultats en dim2 ou 3

# visualize(final_features, dim=2)