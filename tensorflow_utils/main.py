# coding: utf-8

import os
import tensorflow as tf
import argparse

parse = argparse.ArguementParse()
parse.add_argument('--epochs', type=int, help="numbers of traing", default=100)
args = parse.parse_args()

os.path.join()
log_dir = "./logs"
save_dir = "./model"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
    
def test(config):
    
    ## load test data
    print("loading model...")
    
    model = Model(config, is_train=False)
    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
        sv.saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
    
def train(config):
    
    ## load data
    
    print("building model...")
    train_dataset = 
    dev_dataset = 
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
                                                    )
    
    train_iterator = train_dataset.make_one_shot_iterator()
    dev_iterator = dev_dataset.make_one_shot_iterator()
    
    model = Model(config, iterator, word_mat, is_train=True)
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    
    loss_save = 1000.0
    patience = config.patience
    lr = config.init_lr
    
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(config.log_dir)
        saver = tf.train.saver()
        
        train_handle = sess.run(train_iterator.string_handle())
        dev_handle = sess.run(dev_iterator_string_handle())
        sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
        sess.run(tf.assing(model.lr, tf.constant(lr, dtype=tf.float32)))
        
        for ep in tqdm(range(1, config.num_epochs + 1)):
            global_step = sess.run(model.global_step)
            loss, train_op = sess.run([model.loss, model.train_op], feed_dict)
            writer.add_summary()
            
            if global_step % config.period == 0:
                
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
                    
                    
                saver.save(sess, filename)
    
    
if __name__ == '__main__':
    if args.mode = "train":
        print("Traing model")
        train(args)
    elif args.mode.lower() == "test":
        print("Testing on test set...")
        test(args)
    else:
        print("Invalid mode.")
    

