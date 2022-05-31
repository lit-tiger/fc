import os
import sys
sys.path.append('E:\\feb\\feb\\test\\test\\test')
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as py
import matplotlib.pyplot as plt
import numpy as np
from DatasSet import dataset
from Clients import clients
from constant import constant


if __name__ == '__main__':

    noise = constant.NOISE_FLAG

    order = list(range(constant.CLIENT_NUMBER))
    if constant.POOR_QUALITY == 1:
        poor_qiality_dataset = dataset(constant.DATA_NAME, 0)
        trust_client = clients(poor_qiality_dataset.train_data_client[0],poor_qiality_dataset.train_label_client[0], -2, 0,constant.MODEL_NAME)

    dataset = dataset(constant.DATA_NAME, constant.IS_NON_IID)
    
    server = clients(dataset.train_data,dataset.train_label, -1, constant.IS_NON_IID, constant.MODEL_NAME)
    server.model.model.summary()
    #for i in range(len(server.model.model.get_weights())):
        #print(server.model.model.get_weights()[i].shape)
    tf.disable_eager_execution()
    list = None
    step = 0
    norm = 0
    norm_flag = []

    node_clients = []
    for i in range(constant.CLIENT_NUMBER):
        client = clients(dataset.train_data_client[i],dataset.train_label_client[i], i, constant.IS_NON_IID,constant.MODEL_NAME)
        if i%2 == 0:
            client.poor_quilty()
        node_clients.append(client)
    client_loss = np.zeros(constant.TRAIN_NODE_NUMBER)
    #x_batch_max, y_batch_max = node_clients[order[i]].next_batch(constant.NODE_TRAIN_NUMBER*constant.NODE_TRAIN_BATCH_SIZE)
    #server.model.model.fit(x_batch_max, y_batch_max,batch_size=constant.NODE_TRAIN_BATCH_SIZE, epochs=constant.TRAIN_EPOCH, verbose=0)
    np.random.shuffle(order)
    norm = []
    norm_flag = 0
    for i in range(constant.TRAIN_NODE_NUMBER):
        node_clients[order[i]].set_weights(server.model.model.get_weights())
        x_batch_max, y_batch_max = node_clients[order[i]].next_batch(constant.NODE_TRAIN_NUMBER*constant.NODE_TRAIN_BATCH_SIZE)
        node_clients[order[i]].model.model.fit(x_batch_max, y_batch_max,
                batch_size=constant.NODE_TRAIN_BATCH_SIZE, epochs=constant.TRAIN_EPOCH, verbose=0)
        if noise == 1:
            norm.append(node_clients[order[i]].get_self_norm())
            #if node_clients[order[i]].get_self_norm() > norm :
                #norm = node_clients[order[i]].get_self_norm()
                #norm_flag = order[i]
    norm.append(0)
    if noise == 1:
        max_norm = np.max(norm)
        max_weights = node_clients[order[norm.index(max_norm)]].model.model.get_weights()
        #max_weights = node_clients[norm_flag].model.model.get_weights()
    else:
        max_weights = None
    for i in range(constant.TRAIN_NODE_NUMBER):            
        if i == 0 :
            weights = node_clients[order[i]].get_weights(noise, max_weights)#累加模型
        else :
            weights_flag = node_clients[order[i]].get_weights(noise, max_weights)
            for j in range(len(weights)):
                weights[j] +=  weights_flag[j]
            #name.append(node_clients[i].name)
            
    #print(name)
    #name.clear
    #平均模型
    for j in range(len(weights)):
                        weights[j] /=  constant.TRAIN_NODE_NUMBER
    server.model.model.set_weights(weights)
    test_loss, test_acc = server.model.model.evaluate(dataset.test_data, dataset.test_label, verbose=0)
    train_loss, train_acc = server.model.model.evaluate(x_batch_max, y_batch_max, verbose=0)
    #print("Iter "+ str(step)+ ", Minibatch Loss= "+ str(train_loss)+ ", Training Accuracy= "+ str(train_acc)+", Testing Accuracy= "+ str(test_acc))
    print(str(step)+ ","+ str(train_loss)+ ","+ str(train_acc)+","+ str(test_acc))
    step += 1
    weight_list = None

    while step <= constant.MAX_TRAIN_NUMBER:
            np.random.shuffle(order)
            norm = []
            norm_flag = 0
            #先分发模型，再训练
            for i in range(constant.TRAIN_NODE_NUMBER):
                node_clients[order[i]].set_weights(weights)
            #初始化服务器模型
            #训练
            for i in range(constant.TRAIN_NODE_NUMBER):
                x_batch_max, y_batch_max = node_clients[order[i]].next_batch(constant.NODE_TRAIN_NUMBER*constant.NODE_TRAIN_BATCH_SIZE)
                node_clients[order[i]].model.model.fit(x_batch_max, y_batch_max,
                       batch_size=constant.NODE_TRAIN_BATCH_SIZE, epochs=constant.TRAIN_EPOCH, verbose=0)
                
                if noise == 1:
                    norm.append(node_clients[order[i]].get_self_norm())
                    #if node_clients[order[i]].get_self_norm() > norm :
                        #norm = node_clients[order[i]].get_self_norm()
                        #norm_flag = order[i]
            norm.append(0)
            if noise == 1:
                max_norm = np.max(norm)
                max_weights = node_clients[order[norm.index(max_norm)]].model.model.get_weights()
                #max_weights = node_clients[norm_flag].model.model.get_weights()
            else:
                max_weights = None
            for i in range(constant.TRAIN_NODE_NUMBER):            
                    if i == 0 :
                        weights = node_clients[order[i]].get_weights(noise, max_weights)#累加模型
                    else :
                        weights_flag = node_clients[order[i]].get_weights(noise, max_weights)
                        for j in range(len(weights)):
                            weights[j] +=  weights_flag[j]
            #name.append(node_clients[i].name)
            """
                if i == 0 :
                    weights = node_clients[order[i]].model.model.get_weights()#累加模型
                else :
                    weights_flag = node_clients[order[i]].model.model.get_weights()
                    for j in range(len(weights)):
                        weights[j] +=  weights_flag[j]
                        """
                #print(node_clients[order[i]].name)
                #name.append(node_clients[i].name)
            #print(name)
            #name.clear
            #平均模型
            for j in range(len(weights)):
                weights[j] /=  constant.TRAIN_NODE_NUMBER
            server.model.model.set_weights(weights)
            #server.model.new_model(sess)

            #server.model.test_model(step, x_batch_max, y_batch_max, dataset.test_data, dataset.test_label, sess)
            if step%constant.DISPLAY_NUMBER == 0:
                x_batch_max, y_batch_max = server.next_batch(constant.NODE_TRAIN_NUMBER*constant.NODE_TRAIN_BATCH_SIZE)
                test_loss, test_acc = server.model.model.evaluate(dataset.test_data, dataset.test_label, verbose=0)
                train_loss, train_acc = server.model.model.evaluate(x_batch_max, y_batch_max, verbose=0)
                #print("Iter "+ str(step)+ ", Minibatch Loss= "+ str(train_loss)+ ", Training Accuracy= "+ str(train_acc)+", Testing Accuracy= "+ str(test_acc))
                print(str(step)+ ","+ str(train_loss)+ ","+ str(train_acc)+","+ str(test_acc))
            step += 1
    print(constant.DATA_NAME)
    print(constant.CLIENT_NUMBER)
    if constant.NOISE_FLAG == 1:
        print("noise")
            
            
    """

    eval_indices = range(0, constant.MAX_TRAIN_NUMBER, constant.DISPLAY_NUMBER)
    plt.plot(eval_indices, server.model.train_loss, 'k-')
    plt.title('Softmax Loss per iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Softmax Loss')
    plt.show()

    plt.plot(eval_indices, server.model.train_acc, 'k-', label = 'Train Set Accuracy')
    plt.plot(eval_indices, server.model.test_acc, 'r--', label = 'Test Set Accuracy')
    plt.title('Train and Test Accuracy')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.legend(loc = 'lower right')
    plt.show()
    """