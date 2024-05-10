import numpy as np
import matplotlib.pyplot as plt

"Machine learning tools"
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import librosa
from scipy.signal import fftconvolve

from classification.utils.utils import accuracy
helico = pickle.load(open("./helicopter_yt.pickle", "rb"))[0]
helico1 = pickle.load(open("./helicopter.pickle", "rb"))[0]
cls_hel =np.array(["helicopter" for _ in range((len(helico)+len(helico1))//2)])
print(cls_hel.shape, helico.shape, helico1.shape)
birds = pickle.load(open("./birds_yt.pickle", "rb"))[0]
birds1 = pickle.load(open("./birds.pickle", "rb"))[0]
cls_b =np.array(["birds" for _ in range((len(birds)+len(birds1))//2)])
fire = pickle.load(open("./fire_yt.pickle", "rb"))[0]
fire1 = pickle.load(open("./fire.pickle", "rb"))[0]
cls_f =np.array(["fire" for _ in range((len(fire)+len(fire1))//2)])
chain = pickle.load(open("./chainsaw_yt.pickle", "rb"))[0]
chain1 = pickle.load(open("./chainsaw.pickle", "rb"))[0]
cls_c =np.array(["chainsaw" for _ in range((len(chain)+len(chain1))//2)])
hand = pickle.load(open("./handsaw_yt.pickle", "rb"))[0]
hand1 = pickle.load(open("./handsaw.pickle", "rb"))[0]
cls_h=np.array(["handsaw" for _ in range((len(hand)+len(hand1))//2)])


dataset = helico
dataset = np.concatenate((dataset,helico1))
dataset = np.concatenate((dataset,birds))
dataset = np.concatenate((dataset,birds1))
dataset = np.concatenate((dataset,fire))
dataset = np.concatenate((dataset,fire1))
dataset = np.concatenate((dataset,chain))
dataset = np.concatenate((dataset,chain1))
dataset = np.concatenate((dataset,hand))
dataset = np.concatenate((dataset,hand1))
labels = ["helicopter", "birds", "fire", "chainsaw", "handsaw"]
data2_list = cls_hel
data2_list = np.concatenate((data2_list,cls_b))
data2_list = np.concatenate((data2_list,cls_f))
data2_list = np.concatenate((data2_list,cls_c))
data2_list = np.concatenate((data2_list,cls_h))

new_dataset  = np.zeros((len(dataset)//2,48))
print(dataset.shape)
print(new_dataset.shape)
for i in range(len(new_dataset)):
        temp = np.zeros((2,24))
        temp[0] = dataset[2*i]
        temp[0] = dataset[2*i+1]
        new_dataset[i] = np.ravel(temp)
X_train, X_test, y_train, y_test = train_test_split(new_dataset, data2_list, test_size=0.3)

n_trees = 100
model = RandomForestClassifier(n_trees)


#model.fit(X_train, y_train)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print(accuracy(prediction, y_test))
pickle.dump(model,open("model_def.pickle","wb"))