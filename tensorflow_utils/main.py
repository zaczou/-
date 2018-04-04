# coding: utf-8

import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import argparse

parse = argparse.ArguementParse()
parse.add_argument('--mode', type=sting, help="mode of the model", default='train')
parse.add_argument('--epochs', type=int, help="numbers of traing", default=100)
parse.add_argument('--log_dir', type=string, help="directory for events", default="./logs")
parse.add_argument('--save_dir', type=string, help="directory for saving model", default="./save")

args = parse.parse_args()

os.path.join()

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
    
def train(config):
    ## load data    
    print("building model...")
    train_dataset = 
    dev_dataset = 
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
    
    train_iterator = train_dataset.make_one_shot_iterator()
    dev_iterator = dev_dataset.make_one_shot_iterator()
    
    model = Model(config, iterator, is_training=True)
    sess_config = tf.ConfigProto(allow_soft_placement=True)   #如果你指定的设备不存在，允许TF自动分配设备
    sess_config.gpu_options.allow_growth = True #使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放
										        #内存，所以会导致碎片    
    loss_save = 1000.0
    patience = config.patience
    lr = config.init_lr
    
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(config.log_dir)
        saver = tf.train.saver()
        
        train_handle = sess.run(train_iterator.string_handle())
        dev_handle = sess.run(dev_iterator.string_handle())
        #sess.run(tf.assign(model.is_training, tf.constant(True, dtype=tf.bool)))
        #sess.run(tf.assing(model.lr, tf.constant(lr, dtype=tf.float32)))
        
        for ep in tqdm(range(1, config.num_epochs + 1)):
            global_step = sess.run(model.global_step)
            loss, train_op = sess.run([model.loss, model.train_op], feed_dict={handle: train_handle})
            
            if global_step % config.period == 0:
                loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/loss", simple_value=loss), ])
                writer.add_summary(loss_sum, global_step)                
                
            if global_step % checkpoint == 0:
                
                if dev_loss < loss_save:
                    loss_save = dev_loss
                    patience = 0
                else:
                    patience += 1
                if patience >= config.patience:
                    lr /= 2.0
                    loss_save = dev_lss
                    patience = 0
                sess.run()
		
                writer.flush()
                filename = os.path.join(config.save_dir, "model_{}.ckpt".format(global_step))
                saver.save(sess, filename)
    

 def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle):
    answer_dict = {}
    losses = []
    for _ in tqdm(range(1, num_batches + 1)):
        qa_id, loss, yp1, yp2, = sess.run(
            [model.qa_id, model.loss, model.yp1, model.yp2], feed_dict={handle: str_handle})
        answer_dict_, _ = convert_tokens(
            eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
        answer_dict.update(answer_dict_)
        losses.append(loss)
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    metrics["loss"] = loss
    loss_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
    f1_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/f1".format(data_type), simple_value=metrics["f1"]), ])
    em_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/em".format(data_type), simple_value=metrics["exact_match"]), ])
    return metrics, [loss_sum, f1_sum, em_sum]   
    
    
def test(config):
    ## load test data
    print("loading model...")
    test_data =   ## tf.data类型
    test_iterator = test_data.make_one_shot_iterator()
	
    model = Model(config, test_iterator, is_training=False)
    sv = tf.train.Supervisor()
    with sv.managed_session() as sess: ##不需要run(tf.global_variables_initializer())
        sv.saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
        
          
if __name__ == '__main__':
    if args.mode = "train":
        print("Traing model")
        train(args)
    elif args.mode.lower() == "test":
        print("Testing on test set...")
        test(args)
    else:
        print("Invalid mode.")
	exit(0)
    

