import numpy as np
def computeCoccurrenceMatrix(fileList):
    labels = []
    for filename in fileList:
        with open(filename, "r") as f:
            for line in f:
                items = line.split()
                label = items[1:]
                label = [int(i) for i in label]
                labels.append(label)
    gt = np.array(labels)
    #print(gt.shape)
    labelNum = gt.shape[1]
    coMatrix = np.zeros([labelNum,labelNum])

    for i in range(14):
        sourGt = gt[:,i]
        coMatrix[i,i] = (gt[gt[:,i] == 1,:].sum(axis=1) == 1).sum()

        for j in range(i+1,14):
            desGt = gt[:,j]
            coMatrix[i,j] = ((sourGt + desGt) == 2).sum()
            coMatrix[j,i] = coMatrix[i,j]

    return coMatrix

if __name__=="__main__":
    # fileList = ["../ChestX-ray14/labels/train_list.txt",
    #             "../ChestX-ray14/labels/test_list.txt",
    #             "../ChestX-ray14/labels/val_list.txt"]
    # file with form:
    #     [image_name, labels(0 1 1 0 0 0 0...)]
    fileList = ["../ChestX-ray14/labels/test_list.txt"]
    coMatrix = computeCoccurrenceMatrix(fileList)

    np.save("coMatrix_test.npy", coMatrix)
