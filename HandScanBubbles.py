import matplotlib.pyplot as plt
from PIL import Image 
import os
import numpy as np

def GetAllTrigImages():
    path = "../bubbleimages/trig/"
    Files = np.array(["{}{}".format(path,file) for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))])
    return Files

def GetRunInfo(File):
        Date = int(File.split("/")[-1].split("_")[0]) #file should be date_run_event
        Run = int(File.split("/")[-1].split("_")[1])
        Event = int("{}{}{}".format(Date, Run,File.split("/")[-1].split("_")[2])) 
        return Date, Run, Event

def GetBubbleCount(Event, BubbleInfo):
    EventID = Event 
    Cut = BubbleInfo[:,0]==EventID
    if(Cut.sum()!=1): #some events have 2 different bubble count entries
        BubbleCount=9999
    else:
        BubbleCount = int(BubbleInfo[Cut,1])
    return BubbleCount, Cut

def CorrectBubbleCount(File, BubbleInfo):
    Event = GetRunInfo(File)[2]
    DefaultBubbleCount, Indecies = GetBubbleCount(Event, BubbleInfo)
    im = Image.open(File)
    Array = np.asarray(im, dtype=int)
    print(DefaultBubbleCount)
    plt.imshow(Array)
    plt.title(DefaultBubbleCount)
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()
    NewCount = input("Bubble Count?")
    CorrectedBubbleCountList = BubbleInfo
    if(NewCount !=''):
        CorrectedBubbleCount = int(NewCount)
    else:
        CorrectedBubbleCount = DefaultBubbleCount
    CorrectedBubbleCountList[Indecies, 1] = CorrectedBubbleCount
    return CorrectedBubbleCountList

Files = GetAllTrigImages()
#should put an if file exists here
BubbleInfo = np.genfromtxt("NewBubCount.txt", delimiter=" ")
StartingNum = 1000
for i in range(StartingNum, len(Files)):
    print("File {}".format(i))
    BubbleInfo = CorrectBubbleCount(Files[i], BubbleInfo)
    np.savetxt("NewBubCount.txt", BubbleInfo)