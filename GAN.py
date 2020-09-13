import matplotlib.pyplot as plt
import numpy as np

import time
from pathlib import Path 

from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator 

targetSize = (300,300)
batchSize = 32


def loadImages(debug=False):
    datagenerator = ImageDataGenerator(rescale=1./255)
    data = datagenerator.flow_from_directory( 
        Path.cwd() / "Images",
        target_size=targetSize,
        batch_size=batchSize,
        color_mode='rgba')
    if(debug): 
        print(data.image_shape)
        x,y = data.next()
        image = x[0]
        plt.imshow(image)
        plt.show()

    return data

class GAN():
    def __init__(self):
        self.img_rows = 300
        self.img_cols = 300
        self.channels = 4
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        optimizer = Adam(lr=0.0002,beta_1=0.5)
        
        # Create the generator, build and compile
        self.generator = self.buildGenerator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # Create input for generator
        z = Input(shape=(100,))
        img = self.generator(z)
        
        # Create the discriminator, build and compile
        self.discriminator = self.buildDiscriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.discriminator.trainable = False
        valid = self.discriminator(img)

        #create combined model, noise => generate image => determines val
        self.combined = Model(z,valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    def buildGenerator(self):
        noise_shape = 100
        model = Sequential(name='Generator')
        model.add(Dense(256,input_dim=noise_shape))
        model.add(LeakyReLU(0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(0.2))
        model.add(Dense(1024))
        model.add(LeakyReLU(0.2))
        model.add(Dense(np.prod(self.img_shape), activation='sigmoid'))
        model.add(Reshape(self.img_shape))
        
        model.summary()
        noise = Input(shape=(noise_shape,))
        img = model(noise)
        print(img)
        return Model(noise, img) 
    def buildDiscriminator(self):
        model = Sequential(name='Discriminator')
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(1024))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))
        model.add(Dense(512))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))
        model.add(Dense(256))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        
        model.summary()
        img = Input(shape=self.img_shape)
        print("Discriminator input" , img)
        validity = model(img)
        print("Discriminator output", validity)
        return Model(img, validity)
    def train(self,epochs, save_frequency=10):
        data = loadImages()
        batchSize_half = int(batchSize/2)

        for ep in range(epochs):
            start = time.time()

            #############################
            #### Train discriminator ####
            #############################
            # Select random bach, half the size
            idx = np.random.randint(0, len(data))
            imgs, label = data[idx]
            imgs = imgs[:int(len(imgs)/2),]
            
            noise = np.random.rand(batchSize_half,100)
            # print(noise.shape)
            # Generate the second half of new images
            gen_imgs = self.generator.predict(noise)

            # Train
            d_loss_real = self.discriminator.train_on_batch(x=imgs, y=np.ones((batchSize_half,1)))
            d_loss_fake = self.discriminator.train_on_batch(x=gen_imgs, y=np.zeros((batchSize_half, 1)))
            d_loss = np.add(d_loss_real, d_loss_fake) /2

            #########################
            #### Train Generator ####
            #########################
            # Generate new noise for input
            c_x = np.random.rand(batchSize, 100)
            # Create y-data, we want the discriminator to predict 1 (true) for our fake images.
            c_y = np.array([1]*batchSize)

            # Train
            g_loss = self.combined.train_on_batch(c_x,c_y)
            # print(noise)

            end = time.time()
            print("time: ", end - start)
            # print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (ep, d_loss[0], 100*d_loss[1], g_loss[0]))
            print("epoch: ", ep)
            if ep % save_frequency == 0:
                self.saveImage(epoch=ep)
    def saveImage(self, epoch):
        noise = np.random.rand(1,100)
        img = self.generator.predict(noise)
        plt.imshow(img[0])
        plt.savefig("Generatad images/monster_%d.png" % epoch)
        plt.close()
def main():
    gan = GAN()
    gan.train(epochs=20, save_frequency=2)

    # gan.generator.predict(np.random.rand(100))

if __name__ == "__main__":
    main()
