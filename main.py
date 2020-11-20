from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

#added collab path to the myGene "drive/My Drive/unet-master/ to the path data/membrane .."
myGene = trainGenerator(2,'/content/drive/My Drive/unet-master/data/membrane/train','img','label',data_gen_args,save_to_dir = None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=300,epochs=10,callbacks=[model_checkpoint])

#added collab path to the myGene "drive/My Drive/unet-master/ to the path data/membrane .."
testGene = testGenerator("/content/drive/My Drive/unet-master/data/membrane/test")

results = model.predict_generator(testGene,4,verbose=1)

#added collab path to the myGene "drive/My Drive/unet-master/ to the path data/membrane .."
saveResult("/content/drive/My Drive/unet-master/data/membrane/test",results)