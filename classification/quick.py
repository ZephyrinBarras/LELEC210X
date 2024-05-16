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

helico = pickle.load(open("../helicopter_yt.pickle", "rb"))[0]
helico1 = pickle.load(open("../helicopter.pickle", "rb"))[0]
cls_hel =np.array(["helicopter" for _ in range((len(helico)+len(helico1))//2)])
print(cls_hel.shape, helico.shape, helico1.shape)
birds = pickle.load(open("../birds_yt.pickle", "rb"))[0]
birds1 = pickle.load(open("../birds.pickle", "rb"))[0]
cls_b =np.array(["birds" for _ in range((len(birds)+len(birds1))//2)])
fire = pickle.load(open("../fire_yt.pickle", "rb"))[0]
fire1 = pickle.load(open("../fire.pickle", "rb"))[0]
cls_f =np.array(["fire" for _ in range((len(fire)+len(fire1))//2)])
chain = pickle.load(open("../chainsaw_yt.pickle", "rb"))[0]
chain1 = pickle.load(open("../chainsaw.pickle", "rb"))[0]
cls_c =np.array(["chainsaw" for _ in range((len(chain)+len(chain1))//2)])
hand = pickle.load(open("../handsaw_yt.pickle", "rb"))[0]
hand1 = pickle.load(open("../handsaw.pickle", "rb"))[0]
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

# n_trees = 100
# model = RandomForestClassifier(n_trees)

n_estimators_begin = 50
n_estimators_end = 5000
n_estimators_step = 250

parameter_begin = 5
parameter_end = 500
parameter_step = 10

parameters_range = np.arange(parameter_begin, parameter_end, parameter_step)  # ICI LA RANGE DU PARAM MELVEC_LENGTH

# La création des matrices a été adaptée à la modification des boucles i et j
accuracy_matrix = np.zeros(len(parameters_range))

for i in parameters_range:

        temp_acc_split = [] # une accuracy par split
        n_splits = 3
        for _ in range(n_splits):
                X_train, X_test, y_train, y_test = train_test_split(new_dataset, data2_list, test_size=0.3)
                # print("Taille de X_test et de y_test : ", len(X_test), len(y_test))
                # print("Taille de X_train et de y_train : ", len(X_train), len(y_train))

                "Modèle"
                model = RandomForestClassifier(n_estimators=i, max_depth=85)

                "Entraîner le modèle et la PCA"
                model.fit(X_train, y_train)
                prediction = model.predict(X_test)

                temp_acc_split.append(accuracy(prediction, y_test))

        mean_acc = np.mean(temp_acc_split)
        accuracy_matrix[(i - parameter_begin) // parameter_step] = mean_acc

        print("parameters : {}, mean accuracy of the {}-splits : {}".format(i, n_splits,100 * mean_acc))
        print("=====================================")

plt.plot(parameters_range, accuracy_matrix)
plt.show()



# #model.fit(X_train, y_train)
# model.fit(X_train, y_train)
# prediction = model.predict(X_test)
# print(accuracy(prediction, y_test))
# pickle.dump(model,open("model_def.pickle","wb"))