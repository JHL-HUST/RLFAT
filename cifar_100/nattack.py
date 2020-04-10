
import tensorflow as tf
import numpy as np
from helpers import *
import time
import json
import sys
import math
from model_bn import Model
import data_input

npop = 300     # population size
sigma = 0.1    # noise standard deviation
alpha = 0.008  # learning rate

boxmin = 0
boxmax = 1
boxplus = (boxmin + boxmax) / 2.
boxmul = (boxmax - boxmin) / 2.

epsi = 0.03

steps = []

def softmax(x):
        return np.divide(np.exp(x),np.sum(np.exp(x),-1,keepdims=True))
def main():
    with open('config.json') as config_file:
        config = json.load(config_file)
    
    model_file = tf.train.latest_checkpoint(config['model_dir'])
    if model_file is None:
        print('No model found')
        sys.exit()


    totalImages = 0
    succImages = 0
    faillist = []

    input_xs = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y_input = tf.placeholder(tf.int64, shape=[None, 100])
    model = Model(input_xs, y_input, mode='eval')
    
    real_logits_pre = model.pre_softmax
    real_logits = tf.nn.softmax(real_logits_pre)
    
    saver = tf.train.Saver()
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver.restore(sess, model_file)
    
    
    start = 0
    end = 1500
    total = 0
    successlist = []
    printlist = []
    attack_start = time.time()
    
    fashion_mnist = data_input.Data(one_hot=False)
    
    for i in range(start, end):
        success = False
        print('evaluating %d of [%d, %d)' % (i, start, end), file=sys.stderr)

        inputs, targets= fashion_mnist.eval_data.xs[i], fashion_mnist.eval_data.ys[i]
        modify = np.random.randn(1,3,32,32) * 0.001

        logits = sess.run(real_logits, feed_dict={input_xs: [inputs]})
        #print(logits)

        if np.argmax(logits) != targets:
            print('skip the wrong example ', i)
            continue
        totalImages += 1
        for runstep in range(200):
            Nsample = np.random.randn(npop, 3,32,32)

            modify_try = modify.repeat(npop,0) + sigma*Nsample

            newimg = torch_arctanh((inputs-boxplus) / boxmul).transpose(2,0,1)

            inputimg = np.tanh(newimg+modify_try) * boxmul + boxplus
            if runstep % 10 == 0:
                realinputimg = np.tanh(newimg+modify) * boxmul + boxplus
                realdist = realinputimg - (np.tanh(newimg) * boxmul + boxplus)
                realclipdist = np.clip(realdist, -epsi, epsi)
                realclipinput = realclipdist + (np.tanh(newimg) * boxmul + boxplus)
                l2real =  np.sum((realclipinput - (np.tanh(newimg) * boxmul + boxplus))**2)**0.5
                #l2real =  np.abs(realclipinput - inputs.numpy())
                #print(inputs.shape)
                outputsreal = sess.run(real_logits, feed_dict={input_xs: realclipinput.transpose(0,2,3,1)})
                #print(outputsreal)

                #print('lireal: ',np.abs(realclipdist).max())
                #print('l2real: '+str(l2real.max()))
                #print(outputsreal)
                if (np.argmax(outputsreal) != targets) and (np.abs(realclipdist).max() <= epsi):
                    succImages += 1
                    success = True
                    #print('clipimage succImages: '+str(succImages)+'  totalImages: '+str(totalImages))
                    #print('lirealsucc: '+str(realclipdist.max()))
                    successlist.append(i)
                    printlist.append(runstep)

                    steps.append(runstep)
#                     imsave(folder+classes[targets[0]]+'_'+str("%06d" % batch_idx)+'.jpg',inputs.transpose(1,2,0))
                    break
            dist = inputimg - (np.tanh(newimg) * boxmul + boxplus)
            clipdist = np.clip(dist, -epsi, epsi)
            clipinput = (clipdist + (np.tanh(newimg) * boxmul + boxplus)).reshape(npop,3,32,32)
            target_onehot =  np.zeros((1,100))


            target_onehot[0][targets]=1.

            outputs = sess.run(real_logits, feed_dict={input_xs: clipinput.transpose(0,2,3,1)})

            target_onehot = target_onehot.repeat(npop,0)



            real = np.log((target_onehot * outputs).sum(1)+1e-30)
            other = np.log(((1. - target_onehot) * outputs - target_onehot * 10000.).max(1)[0]+1e-30)

            loss1 = np.clip(real - other, 0.,1000)

            Reward = 0.5 * loss1

            Reward = -Reward

            A = (Reward - np.mean(Reward)) / (np.std(Reward)+1e-7)


            modify = modify + (alpha/(npop*sigma)) * ((np.dot(Nsample.reshape(npop,-1).T, A)).reshape((3,32,32)))
        if not success:
            faillist.append(i)
            print('failed:',faillist)
        else:
            print('successed:',successlist)
            print('runstep :', printlist)
        print('now id', i)
        print('successed num', len(successlist))
    print('failed num', len(faillist))
    success_rate = succImages/float(totalImages)
    print('attack time : ', time.time()-attack_start,flush=True)
    print('succ rate', success_rate)
    print(model_file)

if __name__ == '__main__':
    main()

