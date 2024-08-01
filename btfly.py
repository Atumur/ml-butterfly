import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# import Deep learning Libraries
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam, Adamax
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, BatchNormalization
from keras import regularizers
from keras.callbacks import EarlyStopping, LearningRateScheduler

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

df_training = pd.read_csv('Training_set.csv')
df_testing = pd.read_csv('Testing_set.csv')

# Смотрим состав свишек
# df_training.info()
# df_testing.info()

def shape_of_df(df, df_name='df'):
    print(f"{df_name} df имеет {df.shape[0]} строк и {df.shape[1]} столбцов")
    
def check_null(df, ds_name='df'):
    print(f"Кол-во пустых значений в {ds_name} df: ")
    print(df.isnull().sum())

def unique_vals(df, ds_name='df'):
    print(f"Кол-во уникальных значений в {ds_name} df: ")
    print(df.nunique())
    
def seperator(sep=50):
    print("-"*sep)
  
def count_plot(x, df, title, xlabel, ylabel, width, height, order = None, rotation=False, palette='winter', hue=None):
    ncount = len(df)
    plt.figure(figsize=(width,height))
    ax = sns.countplot(x = x, palette=palette, order = order, hue=hue)
    plt.title(title, fontsize=20)
    if rotation:
        plt.xticks(rotation = 'vertical')
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.show()
  
def scalar(img):
    return img

tr_gen = ImageDataGenerator(preprocessing_function= scalar, rotation_range=40, width_shift_range=0.3, height_shift_range=0.2,
                            brightness_range=None, shear_range=0.1, zoom_range=0.3, channel_shift_range=0.4)

ts_gen = ImageDataGenerator(preprocessing_function= scalar, rotation_range=40, width_shift_range=0.3, height_shift_range=0.2,
                            brightness_range=None, shear_range=0.1, zoom_range=0.3, channel_shift_range=0.4)

df_training['filename'] = 'Train/' + df_training['filename']
df_testing['filename'] = 'Tast/' + df_testing['filename']

train_size = 0.75
train_df, valid_df = train_test_split(df_training,  train_size= train_size, shuffle= True, random_state= 123)

# shape_of_df(train_df, "Train")
# shape_of_df(valid_df, "Test")

# check_null(train_df, 'Train')
# check_null(valid_df, 'Test')

# unique_vals(train_df, 'Train')
# unique_vals(valid_df, 'Test')

train_order = df_training['label'].value_counts()
print(train_order)

x = train_df['label']
#count_plot(x, train_df, 'Распределение классов в наборе обучающих данных', 'Вид бабочки', 'Частота', 15, 6, order = train_order.index, rotation=True)

valid_order = valid_df['label'].value_counts()
print(valid_order)

x = valid_df['label']
# count_plot(x, valid_df, 'Распределение классов в наборе данных проверки', 'Вид бабочки', 'Частота', 15, 5, order = valid_order.index, rotation=True, palette='autumn')

print(valid_df.head())

# train_df.head()

batch_size = 16
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)

# Recommended : use custom function for test data batch size, else we can use normal batch size.
ts_length = len(df_testing)
test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length%n == 0 and ts_length/n <= 80]))
test_steps = ts_length // test_batch_size

print(ts_length)
print(test_batch_size)
print(test_steps)

print(train_df['filename'][1])
# print(df_training['label'][1])

train_gen = tr_gen.flow_from_dataframe( train_df, x_col= 'filename', y_col= 'label', target_size= img_size,
                                        color_mode= 'rgb', shuffle= True, batch_size= batch_size)

valid_gen = ts_gen.flow_from_dataframe( valid_df, x_col= 'filename', y_col= 'label', target_size= img_size,
                                        color_mode= 'rgb', shuffle= True, batch_size= batch_size)

# Note: we will use custom test_batch_size, and make shuffle= false

# test_gen = ts_gen.flow_from_dataframe( df_testing, x_col= 'filename', target_size= img_size,
#                                        color_mode= 'rgb', shuffle= False, batch_size= test_batch_size)

g_dict = train_gen.class_indices      # defines dictionary {'class': index}
classes = list(g_dict.keys())       # defines list of dictionary's kays (classes), classes names : string
images, labels = next(train_gen)      # get a batch size samples from the generator

plt.figure(figsize= (8, 8))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    image = images[i] / 255       # scales data to range (0 - 255)
    plt.imshow(image)
    index = np.argmax(labels[i])  # get image index
    class_name = classes[index]   # get class of image
    plt.title(class_name, color= 'blue', fontsize= 12)
    plt.axis('off')
plt.show()

# Create Model Structure
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)
class_count = len(list(train_gen.class_indices.keys())) # to define number of classes in dense layer

# create pre-trained model (you can built on pretrained model such as :  efficientnet, VGG , Resnet )
# we will use efficientnetb3 from EfficientNet family.
base_model = tf.keras.applications.efficientnet.EfficientNetB5(include_top= False, weights= "imagenet", 
                                                               input_shape= img_shape, pooling= 'max')
base_model.trainable = False

model = Sequential([
    base_model,
    BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
    Dense(512, activation = 'relu'),
    Dense(256, kernel_regularizer= regularizers.l2(l= 0.016), activity_regularizer= regularizers.l1(0.006),
                bias_regularizer= regularizers.l1(0.006), activation= 'relu'),
    Dropout(rate= 0.45, seed= 123),
    Dense(class_count, activation= 'softmax')
])

model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

model.summary()

early_stopping = EarlyStopping(monitor='val_accuracy', 
                               patience=5, 
                               restore_best_weights=True,
                               mode='max',
                              )

def step_decay(epoch):
    
     initial_lrate = 0.1
     drop = 0.5
     epochs_drop = 10.0
     lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
     return lrate

lr_scheduler = LearningRateScheduler(step_decay)

batch_size = 16   # set batch size for training
epochs = 1000   # number of all epochs in training

history = model.fit(x= train_gen, epochs= epochs, verbose= 1, validation_data= valid_gen, validation_steps= None, 
                    shuffle= False, batch_size = batch_size, callbacks=[early_stopping])

# Define needed variables
tr_acc = history.history['accuracy']
tr_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]
index_acc = np.argmax(val_acc)
acc_highest = val_acc[index_acc]
Epochs = [i+1 for i in range(len(tr_acc))]
loss_label = f'best epoch= {str(index_loss + 1)}'
acc_label = f'best epoch= {str(index_acc + 1)}'

# Plot training history
plt.figure(figsize= (20, 8))
plt.style.use('fivethirtyeight')

plt.subplot(1, 2, 1)
plt.plot(Epochs, tr_loss, 'r', label= 'Training loss')
plt.plot(Epochs, val_loss, 'g', label= 'Validation loss')
plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(Epochs, tr_acc, 'r', label= 'Training Accuracy')
plt.plot(Epochs, val_acc, 'g', label= 'Validation Accuracy')
plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'blue', label= acc_label)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout
plt.show()