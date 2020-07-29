import gzip
import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')
os.sys.path.insert(0, '../..')
os.sys.path.insert(0, '../../..')

import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
from mp.vmp import VMP
from math_tools.process_traj import TrajNormalizer


class CollectHandWriting:
    def __init__(self, figure, axes, num_images, ifname, ofname, traj_dir):
        self.figure = figure
        self.ax = axes[0]
        self.ax1 = axes[1]
        self.press = False
        self.vmp = VMP(dim=2, kernel_num=10)

        self.ifname = ifname
        self.ofname = ofname
        self.traj_dir = traj_dir

        if not os.path.exists(traj_dir):
            try:
                os.makedirs(traj_dir)
            except OSError:
                raise

        self.num_data = num_images
        f = gzip.open('train-images-idx3-ubyte.gz')
        f.read(16)
        image_size = 28
        buf = f.read(image_size * image_size * num_images * 2)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(2 * num_images, image_size, image_size, 1)
        idx = np.random.choice(2 * num_images, num_images, replace=False)
        self.data = data[idx, :, :, :]
        self.id = 0
        self.next()
        self.connect()

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.cidkey = self.figure.canvas.mpl_connect('key_press_event', self.on_key)

    def on_press(self, event):
        x, y = event.xdata, event.ydata
        self.trajpoints.append(np.array([x,y]))
        self.press = True

    def on_motion(self, event):
        if not self.press: return
        x, y = event.xdata, event.ydata
        self.ax.plot(x, y, 'r.')
        self.trajpoints.append(np.array([x,y]))
        self.figure.canvas.draw()

    def on_release(self, event):
        'on release we reset the press data'
        self.press = False
        traj = np.stack(self.trajpoints)
        traj[:,1] = 28 - traj[:,1]
        timestamps = np.linspace(0, 1, np.shape(traj)[0])
        timestamps = np.expand_dims(timestamps, axis=1)
        traj = np.concatenate([timestamps,traj], axis=1)
        # tprocessor = TrajNormalizer(traj)
        # traj = tprocessor.normalize_timestamp()

        self.vmp.train(traj)
        newTraj = self.vmp.roll(traj[0,1:], traj[-1,1:])
        self.ax1.cla()
        self.ax1.set_xlim(0, 28)
        self.ax1.set_ylim(0, 28)
        self.ax1.plot(traj[:,1], traj[:,2], 'k-.')
        self.ax1.plot(newTraj[:,1], newTraj[:,2], 'g-')
        weight = self.vmp.get_flatten_weights()

        with open(self.ifname, "a") as f:
            image = self.image.copy()
            image = np.expand_dims(image.flatten(), axis=1)
            np.savetxt(f, image.transpose(), delimiter=',', fmt='%.3f')
        with open(self.ofname, "a") as f:
            weight = np.expand_dims(weight, axis=1)
            np.savetxt(f, weight.transpose(), delimiter=',', fmt='%.3f')

        traj_fname = self.traj_dir + '/traj_' + str(self.id)
        np.savetxt(traj_fname, traj, delimiter=',', fmt='%.3f')

        if self.id < self.num_data:
            self.next()
        else:
            self.disconnect()

    def on_key(self, event):
        if event.key == 'z':
            self.next()

    def next(self):
        self.ax.cla()
        self.image = np.asarray(self.data[self.id]).squeeze()
        self.ax.imshow(self.image)
        self.trajpoints = []
        msg = 'collect ' + str(self.id) + '-th data'
        self.figure.suptitle(msg)
        plt.draw()
        self.id = self.id + 1

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.figure.canvas.mpl_disconnect(self.cidpress)
        self.figure.canvas.mpl_disconnect(self.cidrelease)
        self.figure.canvas.mpl_disconnect(self.cidmotion)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-n", "--num_data", dest="n_data", type="int", default=5)
    parser.add_option("-i", "--ifname", dest="ifname", type="string", default="handwriting_queries")
    parser.add_option("-o", "--ofname", dest="ofname", type="string", default="handwriting_weights")
    parser.add_option("-t", "--traj_dir", dest="traj_dir", type="string", default="handwriting_rawdata")

    (options, args) = parser.parse_args(sys.argv)
    fig, axes = plt.subplots(1,2)

    collector = CollectHandWriting(fig, axes, options.n_data, options.ifname, options.ofname, options.traj_dir)
    plt.show()




