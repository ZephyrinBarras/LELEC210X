import numpy as np
file_name = input("file>>")
file = open(file_name,"r")

text = file.readlines()
encode = []
compute = []
send = []
format_ = []
tag = []
melvec = []
shift = []
freq = []
moy = []
def trait(texte):
    texte.strip("\n")
    data = texte.split(" ")
    return int(data[-2])
for i in text:
    if "spec compute" in i:
        compute.append(trait(i))
    if "format" in i:
        format_.append(trait(i))
    if "send" in i:
        send.append(trait(i))
    if "encode" in i:
        encode.append(trait(i))
    if "tag generation" in i:
    	tag.append(trait(i))
    if "melvec" in i:
        melvec.append(trait(i))
    if "moy" in i:
        moy.append(trait(i))
    if "shift" in i:
        shift.append(trait(i))
    if "freq" in i:
        freq.append(trait(i))


print(f"freq {np.mean(np.array(freq))}, {np.mean(np.array(freq))/2:.2f}")
print(f"moy {np.mean(np.array(moy))}, {np.mean(np.array(moy))/2:.2f}")
print(f"shift {np.mean(np.array(shift))}, {np.mean(np.array(shift))/2:.2f}")
tot = np.mean(np.array(shift))+np.mean(np.array(moy))+np.mean(np.array(freq))
print(f"tot {tot/2e3*10:.2f}")
print(f"melvec {np.mean(np.array(melvec))}, {np.mean(np.array(melvec))/2e3:.2f}")
print(f"tag {np.mean(np.array(tag))}, {np.mean(np.array(tag))/2e3:.2f}")		
print(f"encode {np.mean(np.array(encode))/2:.2f}")
print(f"send {np.mean(np.array(send))/2e3:.2f}")
print(f"format_ {np.mean(np.array(format_))}")
print(f"compute {np.mean(np.array(compute))}")
b=np.mean(np.array(format_))+np.mean(np.array(compute))
a = b*10+np.mean(np.array(encode))
print(f"total {b/2e3}")
print(f"durée total cycle {(a+np.mean(np.array(send)))/2e3}")
print(f"duty_cycle {a/(a+np.mean(np.array(send)))}")

print(f"total {a:.2f}, {a/2e3:.2f}ms, {b:.2f}, {b/2e3:.2f}ms")


