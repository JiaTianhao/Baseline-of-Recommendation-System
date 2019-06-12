import numpy as np
import matplotlib.pyplot as plt
from evaluation.explicitdata.Metric import Rmse,Mae
import configparser
from data.Dataset import Dataset
#未做交叉验证
class MF_python(object):
    def __init__(self,dataset):

        config=configparser.ConfigParser()
        config.read("conf/MF.properties")
        self.config=dict(config.items("hyperparameters"))
        self.embedding_size=int(self.config["embedding_size"])
        self.isexplicit=str(self.config["isexplicit"])
        self.ispairwise=str(self.config["ispairwise"])
        self.learning_rate=float(self.config["learning_rate"])
        self.lambdaP=float(self.config["lambdaP"])
        self.lambdaQ=float(self.config["lambdaQ"])
        self.gamma=float(self.config["gamma"])
        self.isEarlyStopping=str(self.config["isEarlyStopping"])
        self.maxIter=int(self.config["maxIter"])
        self.min_val=int(self.config["min_val"])
        self.max_val = int(self.config["max_val"])
        self.lossthreshold=float(self.config["lossthreshold"])

        dataset=self.data_type(dataset)
        self.dataset=dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.iter_rmse=[]
        self.iter_mae=[]

    def data_type(self,dataset):#加载显性数据
        if self.isexplicit=='true':
           dataset=dataset.reset(0)#Dataset(参数)
        return dataset

    def init_model(self):
        self.p=np.random.rand(self.num_users,self.embedding_size)/(self.embedding_size**0.5)
        self.q=np.random.rand(self.num_items,self.embedding_size)/(self.embedding_size**0.5)
        self.loss,self.lastloss=0.0,0.0
        self.lastRmse,self.lastMae=10.0,10.0

    def train_model(self):
        self.init_model()
        iteration=0
        p_delta,q_delta=dict(),dict()
        while iteration<self.maxIter:
            self.loss=0
            if self.isexplicit =='true':
                for u,i in self.dataset.trainMatrix.keys:
                    rating=self.dataset.trainMatrix.get((u,i))
                    pred=self.predict(u,i)
                    error=rating-pred
                    self.loss+=error**2
                    p,q=self.p[u],self.q[i]

                    if u not in p_delta:
                        p_delta[u]=np.zeros(self.embedding_size)
                    if i not in q_delta:
                        q_delta[i] = np.zeros(self.embedding_size)

                    p_delta[u] = self.learning_rate * (-error * q + self.lambdaP * p) + self.gamma * p_delta[u]
                    q_delta[i] = self.learning_rate * (-error * p + self.lambdaQ * q) + self.gamma * q_delta[i]
                    self.p[u]-=p_delta[u]
                    self.q[i]-=q_delta[i]
                self.loss += self.lambdaP * (self.p * self.p).sum() + self.lambdaQ * (self.q * self.q).sum()
                iteration+=1
                if  self.isConverged(iteration):
                    iteration=self.maxIter
                    break
            if self.ispairwise==True:
                pass
    def predict_model_explicit(self):
        res=[]
        for u,i in self.dataset.testMatrix.keys:
            rating=self.dataset.testMatrix.get((u,i))
            prediction=self.predict(u,i)

            prediction=self.denormalize(prediction,self.min_val,self.max_val)
            pred=self.checkRatingBoundary(prediction)
            res.append([u,i,rating,pred])

        rmse=Rmse(res)
        mae=Mae(res)
        self.iter_rmse.append(rmse)  # for plot
        self.iter_mae.append(mae)
        return rmse,mae
    def predict_model_impicit(self):
        pass

    def predict(self,user,item):
        return self.p(user).dot(self.q(item))

    def denormalize(self,prediction,min,max):
        return min+prediction*(max-min)
    def checkRatingBoundary(self, prediction):
        prediction =round( min( max( prediction , self.min_val ) , self.max_val ) ,3)
        return prediction
    def isConverged(self, iter):
        from math import isnan
        if isnan(self.loss):
            print(
                'Loss = NaN or Infinity: current settings does not fit the recommender! Change the settings and try again!')
            exit(-1)
        deltaLoss = (self.lastLoss - self.loss)
        rmse, mae = self.predict_model_explicit()

        # early stopping
        if self.isEarlyStopping =='true':
            cond = self.lastRmse < rmse
            if cond:
                print('test rmse increase, so early stopping')
                return cond
            self.lastRmse = rmse
            self.lastMae = mae

        print('%s iteration %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f rmse=%.5f mae=%.5f' % \
              (self.__class__, iter, self.loss, deltaLoss, self.learning_rate, rmse, mae))

        # check if converged
        cond = abs(deltaLoss) < self.lossthreshold
        converged = cond
        # if not converged:
        # 	self.updateLearningRate(iter)
        self.lastLoss = self.loss
        # shuffle(self.dao.trainingData)
        return converged

    def updateLearningRate(self, iter):
        if iter > 1:
            if abs(self.lastLoss) > abs(self.loss):
                self.config.lr *= 1.05
            else:
                self.config.lr *= 0.5
        if self.config.lr > 1:
            self.config.lr = 1

    def show_rmse(self):
        '''
        show figure for rmse and epoch
        '''
        nums = range(len(self.iter_rmse))
        plt.plot(nums, self.iter_rmse, label='RMSE')
        plt.plot(nums, self.iter_mae, label='MAE')
        plt.xlabel('# of epoch')
        plt.ylabel('metric')
        plt.title(self.__class__)
        plt.legend()
        plt.show()
        pass
    def show_loss(self,loss_all,faloss_all):
        '''
        show figure for rmse and epoch
        '''
        nums = range(len(loss_all))
        plt.plot(nums, loss_all, label='front')
        plt.plot(nums, faloss_all, label='rear')
        plt.xlabel('# of epoch')
        plt.ylabel('loss')
        plt.title('loss experiment')
        plt.legend()
        plt.show()
        pass




