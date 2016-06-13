# -*- coding: utf-8 -*-
from Import import *
import MyLib as myl

#%%
#parameter : When use,change this value
data_dir = './image/' #training data path
n_epoch = 50000  #total epoch number 
im_num = 10      #use image number
im_k = 8         #reduction ratio
reset_num = 4    #stored number in LSTM
use_RGB = False
use_GPU = True
use_DEV = 1      #use gpu device number
min_loss = 100
save_rate = 0.5  # past_loss * save_rate > now loss : save model

#debug param
show_interval = 10 #debug-value(exp:time,image) update interval  
show_im_num = [2,4,8,10] #

#when load Model, change line 67()
#%%
x_train = []
train_data = []

#%%
print '\n---lead dataset Image & convert \n'
im_dir,dir_num = myl.ListDir(data_dir)
print dir_num,im_dir

for j in im_dir:
    print data_dir+j
    im = Image.open(data_dir+j)
    data_h,data_w = im.size
    im_size = [data_w/im_k,data_h/im_k]
    im = im.resize((im_size[1],im_size[0]))# h,w
    if use_RGB:
        im = np.asarray(im.convert('RGB'))
    else:
        im = ImageOps.grayscale(im)   
        im = np.asarray(im)       
    x_train.append(im)    
    if use_RGB:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #cv2.imshow("input"+j,im)    
    if len(x_train) > im_num:
        break

#%%
print '\n---Image Nomarization & Reduction'
channel = 0
if use_RGB:
    channel = 3
else :
    channel = 1
x_train = np.array(x_train).astype(np.float32).reshape((len(x_train),channel,im_size[0],im_size[1]))
y_zero = np.zeros((1,channel*2,im_size[0],im_size[1])).astype(np.float32)
N = len(x_train)

#%%
loss_f = 0
out_num = channel
in_data = np.zeros((1,channel*2,im_size[0],im_size[1])).astype(np.float32)
network = myl.net.NetClass(channel*2,8,out_num,in_data)
network.loadModel('./model/'+'model_pred5001')
network.setup(use_GPU,use_DEV)

#%%
print '\n---Train loop'
sum_time = 0
time_count = 0
epoch = 0

out = 0
write_flag = False
while True:
    sum_loss = 0
    start = time.time()
    show_im = [0]*N
    true_im = [0]*N
    reset_flag = True
 
    epoch += 1
    for i in six.moves.range(0, N):
        x_batch = np.expand_dims(x_train[i],1)
        
        if reset_flag:
            network.resetLSTM()
            network.predict(x_batch)
            reset_flag = False
        else :
            network.mode = True
            network.train(x_batch,y_zero) 
            loss = network.loss
            sum_loss += float(loss.data)
            
        if epoch % show_interval == 0:
            if i in show_im_num:
                result = network.Y.data
                result = network.toCPU(result)
                show_im[i] = np.array(result[0]).astype(np.uint8).reshape((im_size[0], im_size[1],channel))
                if use_RGB:
                        show_im[i] = cv2.cvtColor(show_im[i], cv2.COLOR_BGR2RGB)
                cv2.imshow("predict"+str(i),show_im[i])
                true_im[i] = np.array(x_train[i]).astype(np.uint8).reshape((im_size[0], im_size[1],channel))
                if use_RGB:
                        true_im[i] = cv2.cvtColor(true_im[i], cv2.COLOR_BGR2RGB)
                cv2.imshow("true"+str(i),true_im[i])
        
        if i % reset_num == 0:
            reset_flag = True
        
    one_time = time.time()-start
    sum_time += one_time
    time_count += 1
    train_loss = sum_loss/N        
    train_data.append(float(train_loss))
    if epoch % show_interval == 0:
        print '\tepoch :'+str(epoch)
        print '\ttrain mean loss='+str(train_loss)+" next save :"+str(min_loss)
        print '\ttime sum : {}[sec] ave : {}[sec]'.format(sum_time,sum_time/time_count) 
#%%
    key = cv2.waitKey(1)
    if key in [myl.KEY_ESC, ord('q')]:
        break
    
    if min_loss > train_loss and not train_loss == 0 :
        network.saveModel('model/_model_pred'+str(epoch))
        min_loss = train_loss*save_rate   

    if epoch > n_epoch:
        print '\n!test clear! : '+str(train_loss)
        break
#%%
print '\n---Save model\n'
network.saveModel('model/_model_pred'+str(epoch))

print '\n---Save graph\n'
g_size = (8,6)
g_legend = ["train_loss"]
g_title = "Loss of digit recognition. lasl train loss :"+str(train_loss)  
g_file_name = 'graph_train_loss'+str(epoch)+'.png'
myl.showGraphData(train_data,g_size,g_legend,g_title,g_file_name)

cv2.destroyAllWindows()
