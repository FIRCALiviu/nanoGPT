f = open('training.txt','r')

lines = f.read().splitlines()
f.close()
res = filter(lambda x: x[:4]=='iter',lines) # only lines where loss is 
res  = map(lambda x: x.split(' loss ')[1][:6],res) 
res = map(float,res)

floats = list(res)[:6000]
grouped = []
for i in range(600):
    grouped.append(sum(floats[10*i:10*i+10])/10)
import matplotlib.pyplot as plt

plt.xlabel("10 iteratii (10 batchuri)")
plt.ylabel("cross entropy loss")
plt.plot(grouped)
print(grouped[-1]+grouped[-2]+grouped[-3])
plt.show()