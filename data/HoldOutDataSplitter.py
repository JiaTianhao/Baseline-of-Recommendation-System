import scipy.sparse as sp
import numpy as np
import math
from copy import deepcopy
from util.Logger import logger
class HoldOutDataSplitter(object):
    def __init__(self,path,separator,threshold,splitterRatio=[0.8,0.2]):
        self.path =path +".rating"
        self.separator = separator
        self.splitterRatio = splitterRatio
        self.threshold = threshold
        if float(splitterRatio[0])+ float(splitterRatio[1]) != 1.0:
            raise ValueError("please given a correct splitterRatio")
    def load_data_by_user_time(self):
        logger.info("Loading interaction records from %s "%(self.path))
        pos_per_user={}
        num_ratings=0
        num_items=0
        num_users=0
        #user/item {raw id, inner id} map
        userids = {}
        itemids = {}
        # inverse views of userIds, itemIds, 
        idusers = {}
        iditems={}
        with open(self.path, 'r') as f:
            for line in f.readlines():
                useridx, itemidx,rating, time= line.strip().split(self.separator) 
                num_ratings+=1
                if float(rating)>= self.threshold:#当（默认） threshold=0，是0/1矩阵
                    if  itemidx not in itemids:
                        iditems[num_items]=itemidx
                        itemids[itemidx] = num_items
                        num_items+=1
    
                    if useridx not in userids:
                        idusers[num_users]=useridx
                        userids[useridx]=num_users
                        num_users+=1
                        pos_per_user[userids[useridx]]=[]
                    pos_per_user[userids[useridx]].append([itemids[itemidx],1,int(time)])
                else :#当 threshold=5，是rating矩阵
                    if  itemidx not in itemids:
                        iditems[num_items]=itemidx
                        itemids[itemidx] = num_items
                        num_items+=1
    
                    if useridx not in userids:
                        idusers[num_users]=useridx
                        userids[useridx]=num_users
                        num_users+=1
                        pos_per_user[userids[useridx]]=[]
                    pos_per_user[userids[useridx]].append([itemids[itemidx],rating,int(time)])
                # rating_matrix[self.userids[useridx],self.itemids[itemidx]] = rating

        for u in range(num_users):
            pos_per_user[u]=sorted(pos_per_user[u], key=lambda d: d[2])#每个user交互的集合按时间排序
        logger.info("\"num_users\": %d,\"num_items\":%d, \"num_ratings\":%d"%(num_users,num_items,num_ratings))
        userseq = deepcopy(pos_per_user)
        train_dict = {}
        train_matrix = sp.dok_matrix((num_users, num_items), dtype=np.float32)
        test_matrix = sp.dok_matrix((num_users, num_items), dtype=np.float32)
        time_matrix = sp.dok_matrix((num_users, num_items), dtype=np.float32)
        for u in range(num_users):
            num_ratings_by_user = len(pos_per_user[u])
            num_test_ratings = math.floor(float(self.splitterRatio[1])*num_ratings_by_user)
            if len(pos_per_user[u]) > 3 and num_test_ratings >1:
                for _ in range(num_test_ratings):
                    test_item=pos_per_user[u][-1]
                    pos_per_user[u].pop() 
                    test_matrix[u,test_item[0]] = test_item[1] 
                    time_matrix[u,test_item[0]] = test_item[2]
            items = []
            for enlement in pos_per_user[u]:
                items.append(enlement[0])
                train_matrix[u,enlement[0]]=enlement[1]
                time_matrix[u,enlement[0]] = enlement[2]
            train_dict[u]=items
        return train_matrix,train_dict,test_matrix,userseq,userids,itemids,time_matrix
    '''
    train_matrix ,dockmatrix 用来存储row:user,col:打分item和data:分数,{(u,i)：r}
    train_dict,dict,key=u,value=[ratio交互物品集]，{u:[i,i,i]}
    test_matrix同上，最近ratio交互记录，{（u，i）:r}
    time matrix，同上，data是time，全部数据
    userseq，dict，key=u序号,value=[(交互物品，1/r，time)...]
    userids，dict，key=userid,value=user
    itemIDS，dict, key=itemid,value=item
    '''