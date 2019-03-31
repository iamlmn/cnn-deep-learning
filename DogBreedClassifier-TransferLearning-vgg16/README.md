# Bog breed classifier using transfer learning

The code is to detect the dog breed if dog picture is fed , human face detector using Harcascade is used to detect human faces and gives resemblance to which breed the human face looks similar to else says no human face or dog detected .

Model gave a good accracy with Resnet 50  pretrained weights!

Install requiremnets first. 
```conda create --name dog-project python=3.5
activate dog-project
pip install -r requirements/requirements.txt
```

> Download dog data from [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
---
> Download human face data from [here]https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)

### PreTrained weights 
> Weight botteneck weight (npz format) .

> 1 [VGG16](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) 

> 2 [VGG19](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) 

> 3 [ResNet50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) 

> 4 [Inception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) 

> 5 [Xception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) 

### Using pretrained weigths sample code

```bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
train_resnet = bottleneck_features['train']
valid_resnet = bottleneck_features['valid']
test_resnet = bottleneck_features['test']

resnet_model = Sequential()
resnet_model.add(Flatten(input_shape=train_resnet.shape[1:]))
resnet_model.add(Dense(64, activation='relu'))
resnet_model.add(Dense(133, activation='softmax'))
resnet_model.summary()


resnet_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.resnet.hdf5',
                               verbose=1, save_best_only=True)
                                                          

resnet_model.fit(train_resnet, train_targets, 
          validation_data=(valid_resnet, valid_targets),          
          epochs=70, batch_size=20, callbacks=[checkpointer], verbose=1)
          

resnet_model.load_weights('saved_models/weights.best.resnet.hdf5')
resnet_predictions = [np.argmax(resnet_model.predict(np.expand_dims(feature, axis=0))) for feature in test_resnet]
test_accuracy = 100*np.sum(np.array(resnet_predictions)==np.argmax(test_targets, axis=1))/len(resnet_predictions)

print('Test accuracy: %.4f%%' % test_accuracy)
```

#### Steps :
* Load bottle neck weights
* Add a custom classifier layer(s)
* Build,load,test & save model


### To predict:
```bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))   
    #Load the best model    
    resnet_model.load_weights('saved_models/weights.best.resnet.hdf5')    
     # obtain predicted vector     
    predicted_vector = resnet_model.predict(bottleneck_feature)    
    # return dog breed that is predicted by the model    
    output=dog_names[np.argmax(predicted_vector)]
   ```
    
 ###### keras 2.20.0 has got different dimensions   
    
 Using HaarCascade :
 ```face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml') 
  img = cv2.imread(human_files[2])  
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
  faces = face_cascade.detectMultiScale(gray)
  ```
  
 ### Steps involved :
  * Load CascadeClassifier with xml
  * Read image
  * Gray Scale & detect
  
  [Link to harcasecade xmls](https://github.com/opencv/opencv/tree/master/data/haarcascades)
