import matplotlib.pyplot as plt


range_acc = 0
range_loss = 0
range_sentence = 0
acc = []
loss = []
sentence = []

with open("engine/model/acc.txt", "r") as f:
    for i in f:
        acc.append(float(i))
        range_acc += 1

with open("engine/model/loss.txt", "r") as f:
    for i in f:
        loss.append(float(i))
        range_loss += 1

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(range_acc), acc, label='Точность на обучении')
plt.legend(loc='lower right')
plt.title('Точность')

plt.subplot(1, 2, 2)
plt.plot(range(range_loss), loss, label='Потери на обучении')
plt.legend(loc='upper right')
plt.title('Потери')
plt.savefig('./acc_loss.png')
plt.show()