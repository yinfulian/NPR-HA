# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 23:07:09 2020

@author: 10460
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 23:17:37 2020

@author: 10460
"""
import os
#os.chdir('D:/wyy/recommend/模型实现/')
os.chdir('E:/0-研究生/1-推荐相关/论文投稿/2-节目名称+标签推荐/1-程序和结果/自己模型/Our model/')
import csv

def read_data_csv(txt_file):
    user_data = []
    f=open(txt_file,'r',encoding='utf-8').readlines()
    for line in f:       
        line_split=line.split('#')[:-1]
        line_split=line_split+[line.split('#')[-1].split()]
        user_data.append(line_split)
    return user_data
    
#节目数据表示
def program_label(programs,all_data,MAX_BODY_LENGTH,MAX_LABEL_LENGTH,PD_LENGTH):
    program_dict={}
    word_list=[]
    for name in programs:
        labels=[l for line in all_data for l in line[6] if line[5]==name and l!='空' and l!='']
        labels=list(set(labels))
        pds=[line[4] for line in all_data if line[5]==name]
        pd_freq=Counter(pds)
        pd_freq=sorted(dict(pd_freq).items(),key=lambda v:v[1],reverse=True)
        pd_inf_list=[line[0] for line in pd_freq]

        if len(pd_freq)>=PD_LENGTH:
            pd_inf=pd_inf_list[:PD_LENGTH]
        else:pd_inf=pd_inf_list+['空']*(PD_LENGTH-len(pd_inf_list))
        
        if name not in program_dict:           
            name_split=' '.join(jieba.cut(name)).split()            
            if len(name_split)>=MAX_BODY_LENGTH:
                name_split=name_split[:MAX_BODY_LENGTH]
            else:name_split=name_split+['空']*(MAX_BODY_LENGTH-len(name_split))
            
            if len(labels)>=MAX_LABEL_LENGTH:
                label_list=labels[:MAX_LABEL_LENGTH]              
            else:label_list=labels+['空']*(MAX_LABEL_LENGTH-len(labels))
            
            program_dict[name]=[name,name_split,label_list,pd_inf]
        else:pass
        for w in name_split+label_list:            
            word_list.append(w)
    return program_dict,list(set(word_list))
def save_dict():
    with open('program_dict.txt','w',encoding='utf-8') as f:
        for i in program_dict:
            wirte_str=program_dict[i]
            f.write(str(wirte_str)+'\n')
            
    f.close()
            
def read_txt():
    f=open('./program_dict_2leilabel.txt','r',encoding='utf-8').readlines()
    programs_dict={}
    word_list=[]
    for line in f:
        new_1=line.split('[')[1:]
        name=eval(new_1[0].split(',')[0])
        name_fenci=[eval(n) for n in new_1[1][:-3].split(',')]
        label_2lei=[eval(n) for n in new_1[2][:-3].split(',')]
        pd=[eval(n) for n in new_1[3][:-3].split(',')]
        programs_dict[name]=[name,name_fenci,label_2lei,pd]
        for w in name_fenci+label_2lei:            
            word_list.append(w)
    return programs_dict,word_list    

def read_vector_model(word_embedding_file):
    wordVecDict = {}
    file = open(word_embedding_file,encoding='utf-8')    
    tmp = file.readline()
    num = 0
    for line in file.readlines():
        num += 1
        if num % 5000 == 0:
            print("Reading the %d-th word" % num)
   
        items = line.strip().split()
        if len(items)>2:
            word = items[0]
            try:
                vec = list(map(float, items[1:]))                
                wordVecDict[word]=vec
            except ValueError:
                continue
        else:pass
    file.close()    
    return wordVecDict 
    
def word_dict(wordVecDict,Vocabs,embedding_dim):
    '''词汇表
    将结果编码成数值形式'''
    
    #词汇筛选
    print('num of vocabulary:',len(Vocabs))    
    #获得词汇
    embedding_matrix=[0]*len(Vocabs)   
    #获得词汇转数值编码和数值转词汇编码
    word2idx={}
    idx2word={}
    for i, w in enumerate(Vocabs):
        word2idx[w]=i
        idx2word[i]=w 
           
    for w in word2idx:
        i=word2idx[w]
        if w in wordVecDict:
            embedding_matrix[i]=np.array(wordVecDict[w],dtype='float32')
        else:embedding_matrix[i]=np.zeros(embedding_dim,dtype='float32')  
    embedding_matrix=np.array(embedding_matrix,dtype='float32')
    print(embedding_matrix.shape)
    return word2idx,embedding_matrix
    
#获得训练测试数据集--确定每一个用户分配多少个已观看的节目
def user_watched_data_process(mouth_data,MAX_SENTS):
    user_program_data_all={}    
    user_program_data={} 
    #每一个用户的训练收视节目数据
    min_freq_train=[]
    for i in range(110):
        user_one_data=[line for line in mouth_data if line[0]==str(i)]
        user_program_freq={}
        for line in user_one_data:
            if line[5] in user_program_freq:
                user_program_freq[line[5]]=user_program_freq[line[5]]+1
            else:user_program_freq[line[5]]=1
        user_program_data_all[i]=sorted(dict(user_program_freq).items(),key=lambda v:v[1],reverse=True)
        min_freq_train.append(len(user_program_data_all[i]))
        #print(len(user_program_data_all[i]))
        #选择用于训练用户表示的节目数
        user_program_data[i]=[m[0] for m in user_program_data_all[i][:MAX_SENTS]]
        user_program_data_all[i]= [m[0] for m in user_program_data_all[i][:]]

    return user_program_data,user_program_data_all

#构造随机训练数据
#每一个用户构造20条，每一条包括1个看过，4个没看过

def user_train_data_process(user_program_data,user_program_data_all,program_dict,word2idx,pd2id,name2id,label2id,n):
    Titles_bufenci=[]
    Titles_fenci=[]
    Labels_2lei=[]
    Labels_11lei=[]
    Pds=[]
    User_Titles_bufenci=[]
    User_Titles_fenci=[]
    User_Labels_2lei=[]
    User_Labels_11lei=[]
    User_program_pd=[]
    y_label=[]
    for i in range(110):
        
        #chosen_data=random.sample(user_program_data[i], int(choose_time))
        chosen_data=user_program_data[i][:choose_time] #每个用户的前choosetime条数据
        un_chosen_data_list=[p for p in program_dict if p not in user_program_data_all[i]] #没看过的节目
        #每一个用户选择choose_time个观看节目
        for j in chosen_data:
            new_jlist=[]
            new_jlist.append([j,1])
            #每一个节目再选择4个没有观看的节目作为negative
            un_chosen_data=random.sample(un_chosen_data_list, n)
            for m in un_chosen_data:
                new_jlist.append([m,0])
                
            random.shuffle(new_jlist) #随机排序
            houxuan_watched=[k[0] for k in new_jlist]
            houxuan_label=[k[1] for k in new_jlist]

            
            #训练标签
            y_label.append(houxuan_label)
            #候选节目标题、标签词汇表示
            line_program_name_fencidata=[]
            line_program_name_bufencidata=[]
            line_program_label_2leidata=[]
            line_program_label_11leidata=[]
            line_program_pd_data=[]
            for p in houxuan_watched:
                #节目名称
                tt=program_dict[p][0]
                line_program_name_bufencidata.append(name2id[tt])
                
                tt=program_dict[p][1]
                ttt=[word2idx[m] for m in tt]
                line_program_name_fencidata.append(ttt)                
                
                #节目标签
                ll=program_dict[p][2]
                lll=[label2id[m] for m in ll]
                line_program_label_2leidata.append(lll)  
                
                pdpd=program_dict[p][3]
                pdpdpd=[pd2id[m] for m in pdpd]
                line_program_pd_data.append(pdpdpd)

            #用户观看节目标题、标签表示
            used_watched_list=user_program_data[i]
            line_watched_program_name_fencidata=[]
            line_watched_program_name_bufencidata=[]
            line_watched_program_label_2leidata=[]
            line_watched_program_label_11leidata=[]
            line_watched_program_pd_data=[]
            for p in used_watched_list:
                tt=program_dict[p][0]
                line_watched_program_name_bufencidata.append(name2id[tt])
                
                tt=program_dict[p][1]
                ttt=[word2idx[m] for m in tt]
                line_watched_program_name_fencidata.append(ttt)                
                
                #节目标签
                ll=program_dict[p][2]
                lll=[label2id[m] for m in ll]
                line_watched_program_label_2leidata.append(lll)  
                
                pdpd=program_dict[p][3]
                pdpdpd=[pd2id[m] for m in pdpd]
                line_watched_program_pd_data.append(pdpdpd)

                
                
            Titles_bufenci.append(line_program_name_bufencidata)
            Titles_fenci.append(line_program_name_fencidata)
            Labels_2lei.append(line_program_label_2leidata)
            Pds.append(line_program_pd_data)
            User_program_pd.append(line_watched_program_pd_data)
            User_Titles_bufenci.append(line_watched_program_name_bufencidata)
            User_Titles_fenci.append(line_watched_program_name_fencidata)
            User_Labels_2lei.append(line_watched_program_label_2leidata)
    #train_data_sample=np.array(train_data_sample)
    y_label=np.array(y_label,dtype='int32')
    Titles_bufenci=np.array(Titles_bufenci,dtype='int32')   
    Titles_fenci=np.array(Titles_fenci,dtype='int32')
    Labels_2lei=np.array(Labels_2lei,dtype='int32')   
    Pds=np.array(Pds,dtype='int32')   
    User_program_pd=np.array(User_program_pd,dtype='int32')   
    User_Titles_bufenci=np.array(User_Titles_bufenci,dtype='int32')
    User_Titles_fenci=np.array(User_Titles_fenci,dtype='int32')   
    User_Labels_2lei=np.array(User_Labels_2lei,dtype='int32')
    
    return y_label,Titles_bufenci,Titles_fenci,Labels_2lei,Pds,User_program_pd,User_Titles_bufenci,User_Titles_fenci,User_Labels_2lei

#每一个用户            
def user_test_data_process(user_program_data,test_program_list,user_test_program_data_all):
    user_test_Titles_bufenci=[]
    user_test_Titles_fenci=[]

    user_test_Labels_2lei=[]
    user_test_Pds=[]
    user_test_User_Titles_bufenci=[]
    user_test_User_Titles_fenci=[]
    user_test_User_Labels_2lei=[]
    user_test_User_Pds=[]
    user_test_y_label=[]
    all_test_index=[]
    for i in range(110):
        sess_index=[]
        sess_index.append(len(user_test_y_label))
        wathced_data=user_test_program_data_all[i] #测试集中当前用户观看过的节目
        #用户观看节目标题、标签表示
        used_watched_list=user_program_data[i] #训练数据每个用户实际观看的30个节目
        line_watched_program_name_fencidata=[]
        line_watched_program_name_bufencidata=[]
        line_watched_program_label_2leidata=[]
        line_watched_program_label_11leidata=[]
        line_watched_program_pd_data=[]
        for p in used_watched_list:
            tt=program_dict[p][0]
            line_watched_program_name_bufencidata.append(name2id[tt])
            
            tt=program_dict[p][1]
            ttt=[word2idx[m] for m in tt]
            line_watched_program_name_fencidata.append(ttt)                
            
            #节目标签
            ll=program_dict[p][2]
            lll=[label2id[m] for m in ll]
            line_watched_program_label_2leidata.append(lll)  
            
            pdpd=program_dict[p][3]
            pdpdpd=[pd2id[m] for m in pdpd]
            line_watched_program_pd_data.append(pdpdpd)       
            
        for p in test_program_list: #测试数据中所有用户观看的所有节目
            #训练标签
            if p in wathced_data:
                user_test_y_label.append(1)
            else:
                user_test_y_label.append(0)

            #候选节目标题、标签词汇表示
            tt=program_dict[p][0]
            user_test_Titles_bufenci.append(name2id[tt])
            
            tt=program_dict[p][1]
            ttt=[word2idx[m] for m in tt]
            user_test_Titles_fenci.append(ttt)                
            
            #节目标签
            ll=program_dict[p][2]
            lll=[label2id[m] for m in ll]
            user_test_Labels_2lei.append(lll)  
            
            pdpd=program_dict[p][3]
            pdpdpd=[pd2id[m] for m in pdpd]
            line_watched_program_pd_data.append(pdpdpd)     
            user_test_Pds.append(pdpdpd)


            user_test_User_Titles_bufenci.append(line_watched_program_name_bufencidata)
            user_test_User_Titles_fenci.append(line_watched_program_name_fencidata)
            user_test_User_Labels_2lei.append(line_watched_program_label_2leidata) 
            user_test_User_Pds.append(line_watched_program_pd_data) 
            
        sess_index.append(len(user_test_y_label))              
        all_test_index.append(sess_index)
        
    user_test_y_label=np.array(user_test_y_label,dtype='int32')
    user_test_Titles_bufenci=np.array(user_test_Titles_bufenci,dtype='int32')   
    user_test_Titles_fenci=np.array(user_test_Titles_fenci,dtype='int32')  
    user_test_Labels_2lei=np.array(user_test_Labels_2lei,dtype='int32') 
    user_test_Pds=np.array(user_test_Pds,dtype='int32')  
    user_test_User_Pds=np.array(user_test_User_Pds,dtype='int32')
    user_test_User_Titles_bufenci=np.array(user_test_User_Titles_bufenci,dtype='int32')   
    user_test_User_Titles_fenci=np.array(user_test_User_Titles_fenci,dtype='int32')
    user_test_User_Labels_2lei=np.array(user_test_User_Labels_2lei,dtype='int32')
   
    return all_test_index,user_test_y_label,user_test_Titles_bufenci,user_test_Titles_fenci,user_test_Labels_2lei,user_test_Pds,user_test_User_Pds,user_test_User_Titles_bufenci,user_test_User_Titles_fenci,user_test_User_Labels_2lei

    

def generate_batch_data_train(Titles,Labels,Pds,User_Titles,User_Labels,y_label,User_program_pd,batch_size):
    inputid = np.arange(len(y_label))
    np.random.shuffle(inputid)
    y=y_label
    batches = [inputid[range(batch_size*i, min(len(y), batch_size*(i+1)))] for i in range(len(y)//batch_size+1)]#inputid的索引 每组batch_size个数

    while (True):
        for i in batches:
            candidates_title=Titles[i]
            candidates_title_split=[candidates_title[:,k,:] for k in range(candidates_title.shape[1])]
            candidates_label=Labels[i]
            candidates_label_split=[candidates_label[:,k,:] for k in range(candidates_label.shape[1])]
            
            candidates_Pds=Pds[i]
            candidates_pds_split=[candidates_Pds[:,k,:] for k in range(candidates_Pds.shape[1])]

            watched_User_Titles=User_Titles[i]
            watched_User_Titles_split=[watched_User_Titles[:,k,:] for k in range(watched_User_Titles.shape[1])]

            watched_User_Labels=User_Labels[i]
            watched_User_Labels_split=[watched_User_Labels[:,k,:] for k in range(watched_User_Labels.shape[1])]
            
            watched_User_pds=User_program_pd[i]
            watched_User_pds_split=[watched_User_pds[:,k,:] for k in range(watched_User_pds.shape[1])]

            label=y_label[i]
            yield (candidates_title_split +watched_User_Titles_split+candidates_label_split+watched_User_Labels_split+candidates_pds_split+watched_User_pds_split, label)        
        
def generate_batch_data_test(test_Titles,test_Labels,test_User_Titles,test_User_Labels,user_test_User_Pds,user_test_Pds,test_y_label,batch_size):
    inputid = np.arange(len(test_y_label))
    y=test_y_label
    batches = [inputid[range(batch_size*i, min(len(y), batch_size*(i+1)))] for i in range(len(y)//batch_size+1)]

    while (True):
        for i in batches:
            candidates_title=test_Titles[i]

            candidates_label=test_Labels[i]

            candidates_pd=user_test_Pds[i]

            watched_User_Titles=test_User_Titles[i]
            watched_User_Titles_split=[watched_User_Titles[:,k,:] for k in range(watched_User_Titles.shape[1])]

            watched_User_Labels=test_User_Labels[i]
            watched_User_Labels_split=[watched_User_Labels[:,k,:] for k in range(watched_User_Labels.shape[1])]
            
            watched_User_pds=user_test_User_Pds[i]
            watched_User_pds_split=[watched_User_pds[:,k,:] for k in range(watched_User_pds.shape[1])]

            label=test_y_label[i]

            yield ([candidates_title]+ watched_User_Titles_split+[candidates_label]+watched_User_Labels_split+[candidates_pd]+watched_User_pds_split, label)

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    #argsort()是将X中的元素从小到大排序后，提取对应的索引index，然后输出到y
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

def hr_score(y_true, y_score, k=10):
    #argsort()是将X中的元素从小到大排序后，提取对应的索引index，然后输出到y
    k=sum(y_true)
    zorder = np.argsort(y_score)[::-1]
    zy_true = np.take(y_true, zorder[:k])
    hr_score = sum(zy_true) / sum(y_true)
    return hr_score