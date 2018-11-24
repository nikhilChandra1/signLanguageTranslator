from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
#model_vgg16_conv.summary()
    
    #Create your own input format
keras_input = Input(shape=(224, 224, 3), name = 'image_input')
    
    #Use the generated model 
output_vgg16_conv = model_vgg16_conv(keras_input)
    
    #Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation= 'relu', name='fc1')(x)
x = Dense(4096, activation= 'relu', name='fc2')(x)
x = Dense(30, activation='softmax', name='predictions')(x)
    
    #Create your own model 
classifier = Model(inputs=keras_input, outputs=x)
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.25)

train_generator = train_datagen.flow_from_directory(
    './top30Classes',
    target_size=(224, 224),
    batch_size= 32,
    class_mode='categorical',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    './top30Classes',
    target_size=(224, 224),
    batch_size= 32,
    class_mode='categorical',
    subset='validation')

filePath="/home/aslt/modelFiles/callbacksVGGNetWeights.hdf5"
checkpoint = ModelCheckpoint(filePath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
classifier.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples,
    epochs = 10,
    callbacks = [checkpoint])



classifier.save('VGG16Model.h5')
