import tensorflow as tf
import numpy as np

## MPG:
class basicModel:
    def __init__(self, batch_size, using_batch_norm, seed, eps):
        self.batch_size = batch_size
        self.using_batch_norm = using_batch_norm
        self.seed = seed
        self.global_step = 1
        self.eps = eps

    def load(self, sess, saver, checkpoint_dir, model_dir):
        import re
        import os
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def save(self, sess, saver, checkpoint_dir, model_dir, model_name):
        import os
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=self.global_step)


    def next_batch(self, size):
        idx = np.arange(0, size)
        np.random.shuffle(idx)
        idx = idx[:self.batch_size]
        return idx
