import time
import torch
from ..settings import configs
from random import randrange
import os
from os import walk

class Timer:
    def __init__(self):
        self.ini = time.time()
        self.last = 0
        self.curr = 0
        
    def timeit(self)->float:
        if self.last == 0 and self.curr == 0:
            self.last = time.time()
            self.curr = time.time()
            return 0, 0
        else:
            self.last = self.curr
            self.curr = time.time()
            return time.strftime("%H:%M:%S",time.gmtime(round(self.curr - self.last, 2))), time.strftime("%H:%M:%S",time.gmtime(round(self.curr - self.ini, 2)))


def eval_total(model, testloader, timer, epoch=-1):
    # Only neccessary to evaluate model on one gpu
    if configs.LOCAL_RANK != 0:
        return
    device = configs.DEVICE
    model.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the
    # gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images.to(device))
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.cpu().data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    save_model = 100 * correct / total >= configs.MODEL_SAVE_THRESHOLD
    print(f"{'''''' if epoch==-1 else '''Epoch ''' + str(epoch) + ''': '''}Accuracy of the network on the {total} test images: {100 * correct / float(total)} % ({'saved' if save_model else 'discarded'})")
    t = timer.timeit()
    print(f"Delta time: {t[0]}, Already: {t[1]}\n")
    model.train()
    if save_model:
        if configs.DDP_ON:
            torch.save(model.module.state_dict(), configs.MODEL_DIR + f"{100 * correct / total}".replace('.', '_') + '.pth')
        else:
            torch.save(model.state_dict(), configs.MODEL_DIR + f"{100 * correct / total}".replace('.', '_') + '.pth')


def find_best3(local_rank, rand=False):
    files = next(walk(configs.MODEL_DIR), (None, None, []))[2]
    if len(files) == 0:
        return ''
    acc = sorted([float(i.split('.')[0].replace('_', '.')) for i in files], reverse=True)
    best_acc = acc[:3]
    
    for i in acc[3:]:
        try:
            os.remove(configs.MODEL_DIR + "/" + str(i).replace('.', '_') + ".pth")
        except:
            continue
            
        
    model_name = str(best_acc[randrange(3) if (rand and len(acc[:3]) == 3) else 0]).replace('.', '_') + ".pth"
    if local_rank == 0:
        print(f"Loading one of top 3 best model: {model_name}\n")
    return "/" + model_name


def remove_bad_models():
    files = next(walk(configs.MODEL_DIR), (None, None, []))[2]
    if len(files) == 0:
        return
    acc = sorted([float(i.split('.')[0].replace('_', '.')) for i in files], reverse=True)
    for i in acc[3:]:
        try:
            os.remove(configs.MODEL_DIR + "/" + str(i).replace('.', '_') + ".pth")
        except:
            continue
