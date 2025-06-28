import cv2
import numpy as np
from sklearn.cluster import KMeans

def Coslength(p,q, alpha):
    p = np.power(p, alpha)
    q = np.power(q, alpha)
    p = [i / np.sqrt(np.sum(np.power(p,2))) for i in p]
    q = [i / np.sqrt(np.sum(np.power(q,2))) for i in q]
    r = np.dot(p,q)/(np.linalg.norm(p)*(np.linalg.norm(q)))
    return r

def sift_discriptor(img):
    sift = cv2.xfeatures2d.SIFT_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp = sift.detect(gray, None)
    sift_val_back = sift.compute(gray, kp)[1]
    return sift_val_back

def similarity(sift_val_back1, sift_val_back2, SeedNum):
    [r1, c1] = sift_val_back1.shape
    [r2, c2] = sift_val_back2.shape
    combine = np.vstack((sift_val_back1, sift_val_back2))
    distance = KMeans(n_clusters=SeedNum, random_state=0). \
        fit_transform(combine)
    bow = KMeans(n_clusters=SeedNum, random_state=0). \
        fit(combine).predict(combine)
    cord = KMeans(n_clusters=SeedNum, random_state=0). \
        fit(combine).cluster_centers_
    point_num = len(bow)
    center_num = len(cord)
    L1 = [0 for i in range(center_num)]
    L2 = [0 for i in range(center_num)]
    '''
    Normlization
    '''
    for i in range(0, point_num):
        for j in range(center_num):
            if i < r1:
                L1[bow[i]] += distance[i, j]
            else:
                L2[bow[i]] += distance[i, j]
    '''
    Compute Cosine Distance
    '''
    return Coslength(L1, L2, 0.8)


def VLAD(img1,img2,SeedNum):
    """
    :param img1:
    :param img2:
    :param SeedNum:cluster number
    :return: similarity between two pictures
    """
    sift = cv2.xfeatures2d.SIFT_create()

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    '''
    Search Key Points
    '''
    kp1 = sift.detect(gray1, None)
    kp2 = sift.detect(gray2, None)
    '''
    Compute SIFT descriptors
    '''
    sift_val_back1 = sift.compute(gray1, kp1)[1]
    sift_val_back2 = sift.compute(gray2, kp2)[1]

    if sift_val_back1 is None or sift_val_back2 is None:
        return 0

    [r1, c1] = sift_val_back1.shape
    [r2, c2] = sift_val_back2.shape

    combine = np.vstack((sift_val_back1, sift_val_back2))

    if SeedNum > (r1+r2):
        SeedNum = int((r1+r2)/2)

    km = KMeans(n_clusters=SeedNum, random_state=0, n_init='auto').fit(combine)
    distance = km.transform(combine)
    bow = km.labels_
    cord = km.cluster_centers_
    point_num = len(bow)
    center_num = len(cord)
    L1 = [0 for i in range(center_num)]
    L2 = [0 for i in range(center_num)]
    '''
    Normlization
    '''
    for i in range(0, point_num):
        for j in range(center_num):
            if i < r1:
                L1[bow[i]] += distance[i,j]
            else:
                L2[bow[i]] += distance[i,j]
    '''
    Compute Cosine Distance
    '''
    return Coslength(L1, L2, 0.8)

def VLAD_Cost(detection_img, track_img, seednum=5):
    vlad_cost = np.zeros((len(detection_img), len(track_img)))
    for i in range(len(detection_img)):
        for j in range(len(track_img)):
            vlad_cost[i][j] = VLAD(detection_img[i], track_img[j], seednum)
    return vlad_cost

