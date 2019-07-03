import mnist
import numpy as np
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# 28x28 -> 26x26x8
conv = Conv3x3(8)
# 26x26x8 -> 13x13x8
pool = MaxPool2()
# 13x13x8 -> 10
softmax = Softmax(13*13*8, 10)

# Completes a forward pass of the CNN and calculates the accuracy and cross-entropy loss.
def forward(image, label):
    out = conv.forward((image/255) - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)
    
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0
    
    return out, loss, acc

# Completes a full training step on the given image and label.
# Returns the cross-entropy loss and accuracy.
def train(im, label, lr=.005):
    out, loss, acc = forward(im, label)
    
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]
    
    gradient = softmax.backprop(gradient, lr)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, lr)
    
    return loss, acc

print('\n--- Training the CNN ---')
for epoch in range(5):
    print('--- Epoch %d ---' % (epoch + 1))
    
    permutaion = np.random.permutation(len(train_images))
    train_images = train_images[permutaion]
    train_labels = train_labels[permutaion]
    
    loss = 0
    num_correct = 0

    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i > 0 and i%100 == 0:
            print('\r[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%'
                  %(i + 1, loss / 100, num_correct), end = '', flush = True)
            loss = 0
            num_correct = 0
        
        l, acc = train(im, label)
        loss += l
        num_correct += acc

print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0

for im, label in zip(test_images, test_labels):
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)