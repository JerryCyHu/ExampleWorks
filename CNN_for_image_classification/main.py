import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import matplotlib.pyplot as plt

#defining custom dataset class
class My_dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = torch.tensor(self.data[idx], dtype=torch.float32)
        label = self.labels[idx]
        return image.permute(2, 0, 1), label
    
#custom neural network model
class My_NN(nn.Module):
    def __init__(self, num_classes):
        super(My_NN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, stride=1)  #input channels = 3, output channels = 16
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1)
        self.fc1 = nn.Linear(64 * 3 * 3, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 3 * 3)  #flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

#Function to calculate accuracy
def accuracy(predict, true):# calculate accuracy
    predict_indices = torch.argmax(predict, axis=1)
    true = torch.argmax(true, axis = 1)
    correct_predictions = torch.sum(predict_indices == true)
    accuracy = correct_predictions / len(true)
    
    return accuracy

#Function to update learning rate based on loss
def update_learning_rate(old_loss, new_loss, current_lr, increase_factor=1.05, decrease_factor=0.5):
    if new_loss < old_loss:
        new_lr = current_lr * increase_factor
    elif new_loss > old_loss:
        new_lr = current_lr * decrease_factor
    else:
        new_lr = current_lr

    return new_lr

#Custom cross-entropy loss function, added little noise to prevent overflow
def custom_cross_entropy(predictions, targets, epsilon=1e-8):
    log_softmax_predictions = torch.log(predictions + epsilon)
    return torch.mean(-torch.sum(targets * log_softmax_predictions, dim=1))

def plot_metrics(train_loss_list, train_acc_list, test_acc_list):
    epochs = range(1, len(train_loss_list) + 1)
    
    #Plot training loss
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_list, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss versus Epochs')
    plt.legend()
    
    #Plot training and testing accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_list, label='Training Accuracy')
    plt.plot(epochs, test_acc_list, label='Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy versus Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
def calculate_confusion_matrix(true_labels, predicted_labels, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(true_labels, predicted_labels):
        cm[true][pred] += 1
        
    return cm

#train the model
def train():
    device = torch.device('mps' if getattr(torch, 'has_mps', False) else 'cuda' if torch.cuda.is_available() else 'cpu')
    training_data = np.load('training_data.npy')
    training_labels = np.load('training_label.npy')

    training_data = training_data / 255.0
    training_labels = torch.tensor(training_labels).long()  # Ensure labels are torch tensors

    num_classes = 10
    training_labels_one_hot = F.one_hot(training_labels, num_classes=num_classes)

    train_dataset = My_dataset(training_data, training_labels_one_hot)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    ################
    test_data = np.load('test_data.npy')
    test_labels = np.load('test_label.npy')

    test_data = test_data / 255.0
    test_labels = torch.tensor(test_labels).long()
    test_labels_one_hot = F.one_hot(test_labels, num_classes=10)  #10 classes

    test_dataset = My_dataset(test_data, test_labels_one_hot)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=1)
    #################

    model = My_NN(num_classes)
    model = model.to(device)

    # criterion = nn.CrossEntropyLoss()
    l = 0.1
    optimizer = optim.SGD(model.parameters(), lr=l, weight_decay=0.004)

    num_epochs = 10
    acc = 0
    # for epoch in range(num_epochs):
    epoch = -1
    new_loss = 999
    old_loss = 9999

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    while not ((old_loss-new_loss<0.0001 and old_loss-new_loss>0) or  acc>0.85):
        model.train()
        epoch+=1
        running_loss = 0.0
        all_outputs = torch.empty((0, 10))
        all_outputs = all_outputs.to(device)
        all_labels = torch.empty((0, 10))
        all_labels = all_labels.to(device)
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            labels = labels.type(torch.FloatTensor)
            optimizer.zero_grad()

            labels = labels.to(device)
            inputs = inputs.to(device)

            outputs = model(inputs)
            outputs = outputs.to(device)
            # loss = criterion(outputs, labels)
            loss = custom_cross_entropy(outputs, labels)
            loss = loss.to(device)
            loss.backward()
            optimizer.step()

            all_outputs = torch.cat((all_outputs, outputs))
            all_outputs = all_outputs.to(device)
            all_labels = torch.cat((all_labels, labels))
            all_labels = all_labels.to(device)
            if math.isnan(loss.item()):
                print("is nan!")
            running_loss += loss.item()
        acc = accuracy(all_outputs, all_labels)
        old_loss = new_loss
        new_loss = running_loss / len(train_loader)
        l = update_learning_rate(old_loss, new_loss, l)
        print(f'Epoch {epoch + 1}, Loss: {new_loss}, Accuracy: {acc}, Learning Rate: {l}')

        #####test acc
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for data in test_loader:
                test_images, test_labels = data
                test_images = test_images.to(device)
                test_labels = test_labels.type(torch.FloatTensor).to(device)

                test_outputs = model(test_images)
                test_outputs = test_outputs.to(device)
                
                _, test_predicted = torch.max(test_outputs.data, 1)
                _, test_labels_max = torch.max(test_labels.data, 1)
                test_total += test_labels.size(0)
                test_correct += (test_predicted == test_labels_max).sum().item()
        test_acc = test_correct / test_total
        
        ##############
        train_loss_list.append(new_loss)
        train_acc_list.append(acc.item())
        test_acc_list.append(test_acc)
        
        #update learning rate
        for g in optimizer.param_groups:
            g['lr'] = l

        
    plot_metrics(train_loss_list, train_acc_list, test_acc_list)
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for data in train_loader:
            train_images, train_labels = data
            train_images = train_images.to(device)
            train_labels = train_labels.type(torch.FloatTensor)

            train_outputs = model(train_images)
            train_outputs = train_outputs.cpu()

            true_labels = true_labels+torch.argmax(train_labels, 1).tolist()
            predicted_labels = predicted_labels+torch.argmax(train_outputs, 1).tolist()
    num_classes = 10  # Assuming there are 10 classes
    cm = calculate_confusion_matrix(true_labels, predicted_labels, num_classes)
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Training Data')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2. else "black")
    
    plt.tight_layout()
    plt.show()

    model = model.to('cpu')
    torch.save(model.state_dict(), 'CIFAR_classifier.pth')

#testing the trained model
def test():
    test_data = np.load('test_data.npy')
    test_labels = np.load('test_label.npy')

    test_data = test_data / 255.0
    test_labels = torch.tensor(test_labels).long()
    test_labels_one_hot = F.one_hot(test_labels, num_classes=10)  #10 classes

    test_dataset = My_dataset(test_data, test_labels_one_hot)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=1)
    num_classes = 10

    model = My_NN(num_classes)
    model.load_state_dict(torch.load('CIFAR_classifier.pth'))
    # criterion = nn.CrossEntropyLoss()

    #test
    model.eval()

    num_classes = 10
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            _, labels_max = torch.max(labels, 1)
            c = (predicted == labels_max).squeeze()
            for i in range(labels_max.size(0)):
                label = labels_max[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    class_errors = []
    for i in range(num_classes):
        if class_total[i]:
            error = 1 - (class_correct[i] / class_total[i])
            class_errors.append(error)
            print(f'Class {i} Error: {error:.4f}')

    average_classification_error = sum(class_errors) / num_classes
    print(f'Average Classification Error: {average_classification_error:.4f}')

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            
            loss = custom_cross_entropy(outputs, labels.type(torch.FloatTensor))
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            _, labels_max = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == labels_max).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total

    print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy}')


if __name__ == "__main__":
    train()
    test()
