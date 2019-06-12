'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)
@author: wubin
'''
import numpy as np
from time import time
from evaluation.leaveoneout.LeaveOneOutEvaluate import evaluate_by_loo
from evaluation.foldout.FoldOutEvaluate import evaluate_by_foldout
from util.Logger import logger
def test_model(model,dataset,num_thread=10):
    eval_begin = time()
    model_name=str(model.__class__).split(sep=".")[-1].replace("\'>","")
    if hasattr(model,'isexplicit'):
        if model.isexplicit=='true':
            model.predict_model()
            model.show_loss()
            model.show_rmse()
            return
    if dataset.splitter == "loo":
        (hits, ndcgs,aucs) = evaluate_by_loo(model,dataset.testMatrix,dataset.testNegatives,num_thread)
        hr = np.array(hits).mean()
        ndcg = np.array(ndcgs).mean()
        auc = np.array(aucs).mean()
        logger.info(
            "[model=%s]: [Test HR = %.4f, NDCG = %.4f,AUC = %.4f] [Time=%.1fs]" % (model_name,
            hr, ndcg,auc, time() - eval_begin))
        
    else:
        (pres,recs,maps,ndcgs,mrrs) = evaluate_by_foldout(model,dataset.testMatrix,dataset.testNegatives,num_thread)
        Precision = np.array(pres).mean()
        Recall = np.array(recs).mean()
        MAP = np.array(maps).mean()
        NDCG = np.array(ndcgs).mean()
        MRR = np.array(mrrs).mean()    
        logger.info("[model=%s][%.1fs]: [Test Precision = %.4f, Recall= %.4f, MAP= %.4f, NDCG= %.4f, MRR= %.4f][topk=%.4s]"
               %(model_name,time() - eval_begin, Precision, Recall,MAP,NDCG,MRR,model.topK))     