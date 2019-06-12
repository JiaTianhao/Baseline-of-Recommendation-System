import numpy as np
import matplotlib.pyplot as plt
from evaluation.explicitdata.Metric import Rmse,Mae
import configparser
class SVDPP(object):
    def __init__(self,dataset):
        config = configparser.ConfigParser()
        config.read("conf/SVDPP.properties")
        self.config = dict(config.items("hyperparameters"))
        self.embedding_size = int(self.config["embedding_size"])
        self.isexplicit = str(self.config["isexplicit"])
        self.ispairwise = str(self.config["ispairwise"])
        self.learning_rate = float(self.config["learning_rate"])
        self.lambdaP = float(self.config["lambdaP"])
        self.lambdaQ = float(self.config["lambdaQ"])
        self.lambdaY = float(self.config["lambdaY"])
        self.lambdaB = float(self.config["lambdaB"])
        self.gamma = float(self.config["gamma"])
        self.isEarlyStopping = str(self.config["isEarlyStopping"])
        self.maxIter = int(self.config["maxIter"])
        self.min_val = int(self.config["min_val"])
        self.max_val = int(self.config["max_val"])
        self.lossthreshold = float(self.config["lossthreshold"])

        dataset = self.data_type(dataset)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.iter_rmse = []
        self.iter_mae = []

    # 加载显性数据
    def data_type(self,dataset):
        if self.isexplicit=='true':
           dataset=dataset.reset(0)
        return dataset

    def build_graph(self):
        self.P=np.random.rand(self.num_users,self.embedding_size)/(self.embedding_size**0.5)
        self.Q=np.random.rand(self.num_items,self.embedding_size)/(self.embedding_size**0.5)
        self.loss,self.lastloss=0.0,0.0
        self.lastRmse,self.lastMae=10.0,10.0
        self.Bu = np.random.rand(self.num_users) / (self.embedding_size**0.5)  # bias value of user
        self.Bi = np.random.rand(self.num_items) / (self.embedding_size**0.5)  # bias value of item
        self.Y = np.random.rand(self.num_items, self.embedding_size**0.5) / (
                self.embedding_size ** 0.5)  # implicit preference
        self.SY = dict()

    def train_model(self):
        iteration=0
        while iteration<self.maxIter:
            self.loss=0
            if self.isexplicit =='true':
                for u,i in self.dataset.trainMatrix.keys:
                    rating=self.dataset.trainMatrix.get((u,i))
                    pred=self.predict(u,i)
                    error=rating-pred
                    self.loss+=error**2

                    p,q=self.P[u],self.Q[i]
                    nu,sum_y=self.get_sum_y(u)

                    # update latent vectors
                    self.P[u] += self.learning_rate * (error * q - self.lambdaP * p)
                    self.Q[i] += self.learning_rate * (error * (p + sum_y) - self.lambdaQ * q)

                    self.Bu[u] += self.learning_rate * (error - self.lambdaB * self.Bu[u])
                    self.Bi[i] += self.learning_rate * (error - self.lambdaB * self.Bi[i])

                    u_items = self.dataset.trainDict(u)
                    for j in u_items:
                        self.Y[j]=self.learning_rate*(error/np.sqrt(nu)*q-self.lambdaY*self.Y[j])

                self.loss+=self.lambdaP*(self.P*self.P).sum()+self.lambdaQ*(self.Q*self.Q).sum()+\
                    +self.lambdaB*((self.Bi*self.Bi).sum()+(self.Bu*self.Bu).sum())+self.lambdaY*(self.Y*self.Y).sum()
                iteration+=1
                if self.isConverged(iteration):
                    break

    def predict(self,u,i):
        _,sum_y=self.get_sum_y(u)
        total_len=0
        total_rate=0
        for i in self.num_users:
            total_len+=len(self.dataset.trainDict[i])
            for j in self.dataset.trainDict[i]:
                rating=self.dataset.trainMatrix[i][j]
                total_rate+=rating
        if total_len != 0:
            self.global_mean=float(total_rate)/float(total_len)
        return self.Q[i].dot(self.P[u]+sum_y)+self.global_mean+self.Bi[i]+self.Bu[u]

    def get_sum_y(self, u):
        if u in self.SY:
            return self.SY[u]
        u_items = self.self.dataset.trainDict[u]
        nu = len(u_items)
        sum_y = np.zeros(self.embedding_size)
        for j in u_items:
            sum_y += self.Y[j]
        sum_y /= (np.sqrt(nu))
        self.SY[u] = [nu, sum_y]
        return nu, sum_y

    def predict_model_explicit(self):
        res=[]
        for u,i in self.dataset.testMatrix.keys:
            rating=self.dataset.testMatrix.get((u,i))
            prediction=self.predict(u,i)

            pred=self.checkRatingBoundary(prediction)
            res.append([u,i,rating,pred])

        rmse=Rmse(res)
        mae=Mae(res)
        self.iter_rmse.append(rmse)  # for plot
        self.iter_mae.append(mae)
        return rmse,mae

    def checkRatingBoundary(self, prediction):
        prediction =round( min( max( prediction , self.min_val ) , self.max_val ) ,3)
        return prediction

    def predict_model(self):
        rmse,mae=self.iter_rmse[-1],self.iter_mae[-1]
        print("Rmse:%s,Mae:%s"%rmse,mae)

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
                self.learning_rate *= 1.05
            else:
                self.learning_rate *= 0.5
        if self.learning_rate > 1:
            self.learning_rate = 1

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

    def show_loss(self, loss_all, faloss_all):
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



