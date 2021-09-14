#-*- coding:utf-8 -*-
# checkpoint callback 추가 (epoch 마다 weights 저장)
import torch

from transformers import ElectraConfig, ElectraForMaskedLM, ElectraForPreTraining, load_tf_weights_in_electra
from tensorflow.keras.preprocessing import sequence
from transformers import BertTokenizer
import tensorflow as tf
from tensorflow import keras
from transformers import ElectraTokenizer, TFElectraModel, TFElectraForPreTraining
from transformers import ElectraConfig, ElectraForMaskedLM, ElectraForPreTraining, load_tf_weights_in_electra, TFElectraForPreTraining
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import pandas as pd
from tqdm import tqdm

import argparse
import time
import gc
import numpy as np
import collections
import sys

if __name__ == '__main__':
    
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument(
        '-Level', help='insert preprocessing level', default=None, type=str)
    parser.add_argument(
        '-s_year', help='insert start year', default=None, type=int)
    parser.add_argument(
        '-e_year', help='insert end year', default=None, type=int)
    parser.add_argument(
        '-batch_size', help='insert batch_size', default=None, type=int)
    parser.add_argument(
        '-max_length', help='insert max length', default=None, type=int)
    parser.add_argument(
        '-desc', help='insert description to model', default=None, type=str)
    parser.add_argument(
        '-kisti_label', help='insert description to model', default="FALSE", type=str2bool)
    args = parser.parse_args()
    
    print('batch_size : {0} / max_length : {1} / kisti_label : {2}'.format(args.batch_size, args.max_length, str(args.kisti_label)))
    print('Level : {} / Start year : {} / End year : {} / Desc : {}'.format(args.Level, str(args.s_year), str(args.e_year), args.desc))
    
    start = time.time()  # 시작 시간 저장
    
    # Initialise PyTorch model
    config_path = "kipris_base.json"
    ckpt_path = "kipris_base_all/model.ckpt-500000"
    pytorch_dump_path = "dump/pytorch_model.bin" # model save path


    config = ElectraConfig.from_json_file(config_path)
    print("Building PyTorch model from configuration: {}".format(str(config)))

    model = ElectraForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_electra(
        model, config, ckpt_path, "discriminator" # or "generator"
    )

    # Save pytorch-model

    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)

    # electra gelu 함수
    B = tf.keras.backend


    @tf.function(experimental_relax_shapes=True)
    def gelu(x):
        return 0.5 * x * (1 + B.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))

    # tokenizer 호출

    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('vocab.txt', do_lower_case=False)

    model = TFElectraForPreTraining.from_pretrained("dump/", from_pt=True)
    electra_layers = model.get_layer('electra')

    # 데이터 준비
    df_IPCR = pd.read_csv('../../electra/data/original_data/kipris/kipris_IPCR.csv')
    df_CPC = pd.read_csv('../../electra/data/original_data/kipris/kipris_CPC_rev.csv')
    
    df_IPCR_CPC = pd.concat([df_IPCR, df_CPC])
    del df_IPCR # 변수 삭제
    print("df_IPCR 변수 삭제")
    del df_CPC # 변수 삭제
    print("df_CPC 변수 삭제")
    gc.collect() # 메모리 삭제
    print("gc.collect() 실행")

    def return_code(IPCR, CPC): # 4자리 코드 반환
        if IPCR != IPCR:
            return CPC[2:6]
        elif CPC != CPC:
            return IPCR[2:6]

    df_IPCR_CPC['Code'] = df_IPCR_CPC.apply(lambda x: return_code(x['MainIPCR'], x['MainCPC']), axis=1)
    df_IPCR_CPC = df_IPCR_CPC.drop(['MainIPCR', 'FurtherIPCR', 'MainCPC', 'FurtherCPC'], axis=1)
    df_IPCR_CPC = df_IPCR_CPC.reset_index(drop=True)

    def tag_freq(df):
        freq_section = dict()

        try:
            for tag in df['Code']:
                temp_section = tag
                try:
                    freq_section[temp_section] += 1
                except:
                    freq_section[temp_section] = 1
        except:
            pass

        return freq_section

    freq_code = tag_freq(df_IPCR_CPC)
    sort_freq_code = sorted(freq_code.items(), key = lambda item: item[1], reverse=True)

    freq_code = []
    for n, (i, c) in enumerate(sort_freq_code):
        if c < 200:
            break
        print(n, '/', i, '/', c)
        freq_code.append(i)

    df_IPCR_CPC = df_IPCR_CPC[df_IPCR_CPC['Code'].isin(freq_code)]
    df_IPCR_CPC = df_IPCR_CPC.reset_index(drop=True)

    del freq_code # 변수 삭제
    del sort_freq_code # 변수 삭제
    print("freq_code 변수 삭제")
    gc.collect() # 메모리 삭제
    print("gc.collect() 실행")
    
    #################### kisti label 만 사용 #########################
    if args.kisti_label == True:
        kisti_label = pd.read_csv('korscibert/kisti-dataset/cpc_labelmap.tsv', sep='\t') # kisti label 준비
        kisti_label.columns = ['Code']
        df_IPCR_CPC = pd.merge(kisti_label, df_IPCR_CPC, on='Code', how='inner')
    #################### kisti label 만 사용 #########################

    MAX_LENGTH = args.max_length

    col_list = ['제목', '청구항', '요약', '배경기술', '기술분야', '과제의 해결 수단', '발명의 상세한 설명']
#     col_list = ['청구항']
    encode_train_data = []
    y_data = pd.DataFrame()
    for i in tqdm(range(args.s_year, args.e_year+1)):
        file_name = '../../electra/data/original_data/kipris/kipris_content/preprocessing/L' + str(args.Level) + '/pre_kipris_content_' + str(i) + '.csv'
        df_temp = pd.read_csv(file_name)
        df_temp = pd.merge(df_temp, df_IPCR_CPC, on='file_name', how='inner')
        df_temp = df_temp.drop(['file_name', '도면의 간단한 설명', 'i', 'j'], axis=1)
        df_temp = df_temp.fillna('') # nan값을 blank로 수정
        df_temp = df_temp.drop_duplicates() # 중복 삭제
        df_temp = df_temp.reset_index(drop=True)

        x_data = df_temp.iloc[:, :-1]
        y_data = pd.concat([y_data, df_temp.iloc[:, -1]])

        for i in tqdm(range(len(df_temp))):
    #     for i in tqdm(range(3)):
            data = ''
            for j in range(len(col_list)):
                try:
                    if x_data.loc[i][col_list[j]] != x_data.loc[i][col_list[j]]: # nan 값이라면 continue
                        continue
                    if x_data.loc[i][col_list[j]][0] == '[' and x_data.loc[i][col_list[j]][-1] == ']': # 맨 앞에 '['와 맨 뒤에 ']'를 없애줌
                        x_data.loc[i][col_list[j]] = x_data.loc[i][col_list[j]][1:-1]
#                         print('[] 처리')
#                     temp = x_data.loc[i][col_list[j]].replace('u3000', ' ')
                    temp = x_data.loc[i][col_list[j]].replace('u3000', ' ')[:80]
#                     if ' ' in temp and len(temp) > 256:
# #                     if ' ' in temp:
#                         temp = temp[:256]
#                         index = temp.rfind(' ')
#                         temp = temp[:index]
                    data += temp + ' '
        #     data += '\n'
        #     print(data)
                except: # 컬럼 내용이 비어있으면 except 발생함 (= blank이면 except 발생)
    #                 print("except 발생")
    #                 print(i, col_list[j])
                    continue
            data = data[:MAX_LENGTH]
            encode_train_data.append(tokenizer.encode(data, add_special_tokens = True, max_length=MAX_LENGTH, truncation=True))
            
    y_data = y_data.iloc[:][0]
    train_x = sequence.pad_sequences(encode_train_data, maxlen=MAX_LENGTH, value=0)
    print("Original Text : ", x_data)
    del x_data # 변수 삭제
    print("x_data 변수 삭제")
    del encode_train_data # 변수 삭제
    print("encode_train_data 변수 삭제")
    gc.collect() # 메모리 삭제
    print("gc.collect() 실행")

    # category = set(y_data[:3])
    category = set(y_data)
    category_idx = dict()
    for n, cate in enumerate(category):
        category_idx[cate] = n
    idx_train_label = []
    # for l in tqdm(y_data[:3]):
    for l in tqdm(y_data):
        idx_train_label.append(category_idx[l])


    train_y = to_categorical(idx_train_label)
    print(train_x.shape, train_y.shape)
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
    print('Train : {} / {}'.format(train_x.shape, train_y.shape))
    print('Test : {} / {}'.format(test_x.shape, test_y.shape))
    NUM_LABEL = len(train_y[0])
    
    print("데이터 처리 실행 시간 : ", time.time() - start)  # 데이터 처리 실행 시간

    del y_data # y_data 변수 삭제
    print("y_data 변수 삭제")
    gc.collect() # 메모리 삭제
    print("gc.collect() 실행")

    print("Train Labels : {}".format(len(category_idx)))

    # 모델 생성
    input_layer = keras.layers.Input(shape=(MAX_LENGTH,), dtype='int32')
    electra_l = electra_layers(input_layer)[0]


    X = keras.layers.Lambda(lambda seq: seq[:, 0, :])(electra_l)

    X = keras.layers.Dropout(0.1)(X)
    X = keras.layers.Dense(units=256, activation="relu")(X)
    X = keras.layers.Dropout(0.1)(X)

    output_layer = keras.layers.Dense(NUM_LABEL,activation='softmax')(X)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    model.build(input_shape=(None, MAX_LENGTH))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-5),
        loss = keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
    )

    print(model.summary())
    
    checkpoint_path = 'models/model_L{0}_{1}-{2}/cp_L{0}_{1}-{2}_{desc}.ckpt'.format(args.Level, str(args.s_year), str(args.e_year), desc=args.desc)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                   save_weights_only=True, save_best_only=True, save_freq = 'epoch', mode='min',
                                   verbose=1)# Train the model with
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    tf.debugging.set_log_device_placement(True)

    print("fit 시작")
    train_start = time.time()  # 학습 시작 시간 저장
    history = model.fit(train_x, train_y, batch_size=args.batch_size, validation_split=0.1, epochs=10, callbacks=[callback, cp_callback])
        
    ## keras 학습 참고
    # model.fit(train_x, train_y, batch_size=1, validation_split=0.1, epochs=10)

    print("evalutate 시작")
    score_his = model.evaluate(test_x, test_y)
    print("test loss, test acc:", score_his)
    
    y_pred = model.predict(test_x)

    pred_y = []
    for y in y_pred:
    #     tmp = np.zeros(len(categories))
        tmp = np.zeros(len(category_idx))
        tmp[y.argmax()] = 1
        pred_y.append(tmp)


    report_file_name = 'report_L{0}_{1}-{2}_{desc}.txt'.format(args.Level, str(args.s_year), str(args.e_year), desc=args.desc)
    with open(report_file_name, 'w') as f:
#         f.write(classification_report(test_y, np.array(pred_y), target_names=categories.keys()))
        f.write(classification_report(test_y, np.array(pred_y), target_names=category_idx.keys()))

    idx_2_cate = {}
#     for i in categories:
#         idx_2_cate[categories[i]] = i
    for i in category_idx:
        idx_2_cate[category_idx[i]] = i
    
    result_file_name = 'result_L{0}_{1}-{2}_{desc}.txt'.format(args.Level, str(args.s_year), str(args.e_year), desc=args.desc)
    with open(result_file_name, 'w') as f:
        for t, p in zip(test_y, pred_y):
            f.write(idx_2_cate[t.argmax()]+'\t'+ idx_2_cate[p.argmax()]+'\n')
    
    y_preds = np.argsort(y_pred, axis=1)[:,-3:]

    result_top3_file_name = 'result_L{0}_{1}-{2}_{desc}_top-3.txt'.format(args.Level, str(args.s_year), str(args.e_year), desc=args.desc)
    with open(result_top3_file_name, 'w') as f:
        f.write('{:^10}{:^10}{:^10}{:^10}'.format("3순위", "2순위", "1순위", "target")+'\n')
        for i in range(len(test_y)):
        #     print("{:^10}{:^10}{:^10}{:^10}".format(inv_categories[y_preds[i][2]], inv_categories[y_preds[i][1]], inv_categories[y_preds[i][0]], ))
            f.write("{:^10}{:^14}{:^11}{:^11}".format(idx_2_cate[y_preds[i][0]], idx_2_cate[y_preds[i][1]], idx_2_cate[y_preds[i][2]], idx_2_cate[test_y[i].argmax()])+'\n')
    
    print("학습 실행 시간 : ", time.time() - train_start)  # 학습 실행 시간
    
    model_file_name = 'models/model_L{0}_{1}-{2}/model_L{0}_{1}-{2}_{desc}.h5'.format(args.Level, str(args.s_year), str(args.e_year), desc=args.desc)
#     weights_file_name = 'models/model_L{0}_{1}-{2}/ckpt'.format(args.Level, str(args.s_year), str(args.e_year))
    model.save(model_file_name)
#     model.save_weights(weights_file_name)
