import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, Dropout, Dense, Flatten, Activation, BatchNormalization, UpSampling2D, Reshape, LeakyReLU, Input
from keras.datasets import mnist

# print(tf.__version__)
# print(keras.__version__)
if not os.path.exists('./gan_images'):
    os.makedirs('./gan_images')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(3)
# tf.random.set_seed(3)

# 생성자 모델
generator = Sequential([
    Dense(128*7*7, input_dim=100),
    Activation(LeakyReLU(0.2)),
    BatchNormalization(),
    Reshape((7, 7, 128)),
    UpSampling2D(),

    Conv2D(64, 5, padding='same'),
    BatchNormalization(),
    Activation(LeakyReLU(0.2)),
    UpSampling2D(),
    Conv2D(1, 5, padding='same', activation='tanh')
])

# 판별자 모델
discriminator = Sequential([
    Conv2D(64, 5, strides=2, padding='same', input_shape=(28, 28, 1)),
    Activation(LeakyReLU(0.2)),
    Dropout(0.3),
    Conv2D(128, 5, strides=2, padding='same'),
    Activation(LeakyReLU(0.2)),
    Dropout(0.3),
    Flatten(),
    Dense(1, activation='sigmoid')
])

discriminator.compile(loss='binary_crossentropy', optimizer='adam')
discriminator.trainable = False

ginout = Input(shape=(100, ))

dis_output = discriminator(generator(ginout))
gan = Model(ginout, dis_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')
gan.summary()

def gan_train(epoch, batch_size, saving_interval):
    (x_train, _), (_, _) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_train = (x_train - 127.5) / 127.5
    true = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for i in range(epoch):
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        imgs = x_train[idx]
        d_loss_real = discriminator.train_on_batch(imgs, true)

        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(noise)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        g_loss = gan.train_on_batch(noise, true)

        print('epoch:{0:d}, d_loss:{1:.4f}, g_loss:{2:.4f}'.format(i, d_loss, g_loss))

        if i % saving_interval == 0:
            noise = np.random.normal(0, 1, (25, 100))
            gen_imgs = generator.predict(noise)
            gen_imgs = 0.5 * gen_imgs + 0.5

            fig, axs = plt.subplots(5, 5)
            count = 0

            for j in range(5):
                for k in range(5):
                    axs[j, k].imshow(gen_imgs[count, : , :, 0], cmap='gray')
                    axs[j, k].axis('off')
                    count += 1
            
            fig.savefig('gan_images/gan_mnist_%d.png' %i)


gan_train(4001, 32, 200)