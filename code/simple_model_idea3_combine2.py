# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 00:21:36 2020

@author: 10460
"""
import os
os.chdir('E:/0-研究生/1-推荐相关/论文投稿/2-节目名称+标签推荐/1-程序和结果/自己模型/Our model/')
#os.chdir('D:/wyy/recommend/模型实现/')
import jieba
import csv
import numpy as np
import random
from collections import Counter
#用户数据读入
MAX_BODY_LENGTH=6
MAX_LABEL_LENGTH=2
PD_LENGTH=1
MAX_SENTS=30
PD_MAX_SENTS=30
choose_time=20
batch_size=30
n=4

#读入训练和测试数据
train_txt_file5='E:/0-研究生/1-推荐相关/师姐研究/模型实现/data/overall_user_data/train_mouth5_data_300s_2label.txt'
train_txt_file6='data/overall_user_data/train_mouth6_data_300s_2label.txt'
train_txt_file7='data/overall_user_data/train_mouth7_data_300s_2label.txt'
train_txt_file8='data/overall_user_data/train_mouth8_data_300s_2label.txt'
train_txt_file9='data/overall_user_data/train_mouth9_data_300s_2label.txt'
train_txt_file10='data/overall_user_data/train_mouth10_data_300s_2label.txt'
test_txt_file='E:/0-研究生/1-推荐相关/师姐研究/模型实现/data/overall_user_data/test_mouth6_data_60s_2label.txt'

train_mouth_data5=read_data_csv(train_txt_file5)
train_mouth_data6=read_data_csv(train_txt_file6)
train_mouth_data7=read_data_csv(train_txt_file7)
train_mouth_data8=read_data_csv(train_txt_file8)
train_mouth_data9=read_data_csv(train_txt_file9)

train_mouth_data=train_mouth_data5
#train_mouth_data=train_mouth_data5+train_mouth_data6+train_mouth_data7+train_mouth_data8+train_mouth_data9
test_mouth_data=read_data_csv(test_txt_file)


all_data=train_mouth_data+test_mouth_data
print('train data:',len(train_mouth_data)) 

print('test data:',len(test_mouth_data)) 

#频道的规律
#频道信息
pd_list=[]
for line in all_data:   
    pd_list.append(line[4])

Counter(pd_list)
pd_list=list(set(pd_list))
print('pd num:',len(pd_list))
pd2id={}
for index,i in enumerate(pd_list):
    pd2id[i]=index
pd2id['空']=len(pd2id.keys())
    
#标签
label_list=[]
for line in all_data:
    if len(line[-1])>MAX_LABEL_LENGTH:
        for i in line[-1][:MAX_LABEL_LENGTH]:
            label_list.append(i)
    else:
        for i in line[-1]:
            label_list.append(i)


Counter(label_list)
label_list=list(set(label_list))
print('label num:',len(label_list))
label2id={}
for index,i in enumerate(label_list):
    label2id[i]=index
label2id['空']=len(label2id.keys())

#节目名称
program_list=[]
for line in all_data:   
    program_list.append(line[5])

Counter(program_list)
program_list=list(set(program_list))
print('program num:',len(program_list))
name2id={}
for index,i in enumerate(program_list):
    name2id[i]=index
name2id['空']=len(name2id.keys())
  
#处理数据，使得每一个节目对应一个节目名称、节目名称分词、节目标签、节目频道
program_dict,word_list=program_label(name2id,all_data,MAX_BODY_LENGTH,MAX_LABEL_LENGTH,PD_LENGTH)
#save_dict()
programs_dict,word_list=read_txt()



#所有词汇
Vocabs=list(set(word_list))
print('Vocabs num:',len(Vocabs))

#预处理词嵌入读入
embedding_dim=300
#word_embedding_file='D:/wyy/data/pre_model/Sogou_News/word2vec.sohuall.300d.skip.txt' 
word_embedding_file='E:/0-研究生/1-推荐相关/师姐研究/模型实现/data/pre-train-embedding/word2vec.sohuall.300d.skip.txt' 
wordVecDict=read_vector_model(word_embedding_file)
word2idx,embedding_matrix=word_dict(wordVecDict,Vocabs, embedding_dim)

#获得训练和测试数据集，每一个用户对应的全部节目和部分节目
user_program_data,user_program_data_all=user_watched_data_process(train_mouth_data,MAX_SENTS)  
user_test_program_data,user_test_program_data_all=user_watched_data_process(test_mouth_data,10)   


#获得随机训练数据
#每一个用户构造20条，每一条包括1个看过，4个没看过
y_label,Titles_bufenci,Titles_fenci,Labels_2lei,Pds,User_program_pd,User_Titles_bufenci,User_Titles_fenci,User_Labels_2lei=user_train_data_process(user_program_data,user_program_data_all,program_dict,word2idx,pd2id,name2id,label2id,n)

#train_data_sample,y_train_label=user_watched_data_process(word2idx,user_program_data,user_mouth_data,train_program_dict,choose_time)
#train_data_sample=user_watched_data_process(user_program_data,train_program_dict,choose_time)
#获得测试数据集
test_program_list=[]
for line in test_mouth_data:   
    test_program_list.append(line[5])

test_program_list=list(set(test_program_list))
print('program num:',len(test_program_list))

all_test_index,user_test_y_label,user_test_Titles_bufenci,user_test_Titles_fenci,user_test_Labels_2lei,user_test_Pds,user_test_User_Pds,user_test_User_Titles_bufenci,user_test_User_Titles_fenci,user_test_User_Labels_2lei=user_test_data_process(user_program_data,test_program_list,user_test_program_data_all)

#训练过程
import keras
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.optimizers import *

#节目名称表示

title_input = Input(shape=(MAX_BODY_LENGTH,), dtype='int32')

embedding_layer = Embedding(len(Vocabs), embedding_dim, weights=[embedding_matrix],trainable=True)

embedded_sequences_title = embedding_layer(title_input)
embedded_sequences_title=Dropout(0.2)(embedded_sequences_title)


title_cnn = Convolution1D(100, 3,  padding='same', activation='relu', strides=1)(embedded_sequences_title)
title_cnn=Dropout(0.2)(title_cnn)

title_attention = Dense(100,activation='tanh')(title_cnn)
title_attention = Flatten()(Dense(1)(title_attention))
title_attention_weight = Activation('softmax')(title_attention)
title_rep=keras.layers.Dot((1, 1))([title_cnn, title_attention_weight])
 

#节目标签表示

label_input=Input((MAX_LABEL_LENGTH,), dtype='int32') 
embedded_sequences_label = embedding_layer(label_input)
embedded_sequences_label =Dropout(0.2)(embedded_sequences_label)
label_merged =Flatten()(embedded_sequences_label)   
label_rep=Dense(100,activation='relu')(label_merged) 

#节目频道表示
pd_input=Input((1,), dtype='int32') 

pd_embedding_layer = Embedding(len(pd2id), 50,trainable=True)
pd_rep=Dense(100,activation='relu')(Flatten()(pd_embedding_layer(pd_input)))


#节目表示
all_channel=[title_rep,label_rep,pd_rep]
views=concatenate([Lambda(lambda x: K.expand_dims(x,axis=1))(channel) for channel in all_channel],axis=1)

attentionv = Dense(100,activation='tanh')(views)

attention_weightv =Lambda(lambda x:K.squeeze(x,axis=-1))(Dense(1)(attentionv))
attention_weightv =Activation('softmax')(attention_weightv)

programrep=keras.layers.Dot((1, 1))([views, attention_weightv])     
programEncoder = Model([title_input,label_input,pd_input],programrep) 

#用户表示
MAX_SENTS=30

browsed_program_title_input = [keras.Input((MAX_BODY_LENGTH,), dtype='int32') for _ in range(MAX_SENTS)]

browsed_label_input = [keras.Input((MAX_LABEL_LENGTH,), dtype='int32') for _ in range(MAX_SENTS)]
browsed_pd_input = [keras.Input((1,), dtype='int32') for _ in range(MAX_SENTS)]

watched_program = [programEncoder([browsed_program_title_input[_],browsed_label_input[_],browsed_pd_input[_]]) for _ in range(MAX_SENTS)]
watched_programrep =concatenate([Lambda(lambda x: K.expand_dims(x,axis=1))(news) for news in watched_program],axis=1)    

attentionn = Dense(100,activation='tanh')(watched_programrep)
attentionn =Flatten()(Dense(1)(attentionn))
attention_weightn = Activation('softmax')(attentionn)
user_rep=keras.layers.Dot((1, 1))([watched_programrep, attention_weightn])  




#训练数据表示
npratio=4
candidates_title = [keras.Input((MAX_BODY_LENGTH,), dtype='int32') for _ in range(npratio+1)]

candidates_label = [keras.Input((MAX_LABEL_LENGTH,), dtype='int32') for _ in range(npratio+1)]

candidates_pd = [keras.Input((1,), dtype='int32') for _ in range(npratio+1)]

candidate_vecs = [programEncoder([candidates_title[_],candidates_label[_],candidates_pd[_]]) for _ in range(npratio+1)]
#[None,100]*5
logits = [keras.layers.dot([user_rep, candidate_vec], axes=-1) for candidate_vec in candidate_vecs]
#[5,1]
logits = keras.layers.Activation(keras.activations.softmax)(keras.layers.concatenate(logits))
#(None,5)

model = Model(candidates_title+browsed_program_title_input+candidates_label+browsed_label_input+candidates_pd+browsed_pd_input, logits)
#model= keras.Model([candidate_one_title]+[candidate_one_label]+browsed_program_title_input+browsed_label_input, logits)
#model = Model([candidates_title,browsed_program_title_input,candidates_label,browsed_label_input], logits)


#预测数据

candidate_one_title = keras.Input((MAX_BODY_LENGTH,))

candidate_one_label = keras.Input((MAX_LABEL_LENGTH,))

candidate_one_pd= keras.Input((1,))

candidate_one_vec=programEncoder([candidate_one_title,candidate_one_label,candidate_one_pd])

score = keras.layers.Activation(keras.activations.sigmoid)(keras.layers.dot([user_rep, candidate_one_vec], axes=-1))

model_test = keras.Model([candidate_one_title]+browsed_program_title_input+[candidate_one_label]+browsed_label_input+[candidate_one_pd]+browsed_pd_input, score)


model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc'])


results=[]
for ep in range(10):
    traingen=generate_batch_data_train(Titles_fenci,Labels_2lei,Pds,User_Titles_fenci,User_Labels_2lei,y_label,User_program_pd,30)
    model.fit_generator(traingen, epochs=1,steps_per_epoch=len(y_label)//30)
    testgen=generate_batch_data_test(user_test_Titles_fenci,user_test_Labels_2lei,user_test_User_Titles_fenci,user_test_User_Labels_2lei,user_test_User_Pds,user_test_Pds,user_test_y_label, 30)
    click_score = model_test.predict_generator(testgen, steps=len(user_test_y_label)//30,verbose=1)
    from sklearn.metrics import roc_auc_score
    all_auc=[]
    all_mrr=[]
    all_ndcg=[]
    all_ndcg2=[]
    all_hr=[]
    for index,m in enumerate(all_test_index[:109]):
        #print(m)
        all_auc.append(roc_auc_score(user_test_y_label[m[0]:m[1]],click_score[m[0]:m[1],0]))
        all_mrr.append(mrr_score(user_test_y_label[m[0]:m[1]],click_score[m[0]:m[1],0]))
        all_ndcg.append(ndcg_score(user_test_y_label[m[0]:m[1]],click_score[m[0]:m[1],0],k=5))
        all_ndcg2.append(ndcg_score(user_test_y_label[m[0]:m[1]],click_score[m[0]:m[1],0],k=10))
        all_hr.append(hr_score(user_test_y_label[m[0]:m[1]],click_score[m[0]:m[1],0],k=100))
    results.append([np.mean(all_auc),np.mean(all_mrr),np.mean(all_ndcg),np.mean(all_ndcg2),np.mean(all_hr)])
    print(np.mean(all_auc),np.mean(all_mrr),np.mean(all_ndcg),np.mean(all_ndcg2),np.mean(all_hr))

