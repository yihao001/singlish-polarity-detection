"""
Bi-GRU with pre-trained multilingual embedding from BERT
"""

import kashgari
from kashgari.tasks.classification import BiLSTM_Model, BiGRU_Model
from kashgari.embeddings import BERTEmbedding

from kashgari.callbacks import EvalCallBack, KashgariModelCheckpoint
from keras.callbacks import EarlyStopping, ModelCheckpoint

from dataprep import generate_data

import logging
logging.basicConfig(level='DEBUG')


TARGET_DIRECTORY = './'

EPOCH = 100
BATCH_SIZE = 512
PATIENCE = 10

bert_embed = BERTEmbedding(TARGET_DIRECTORY + '/model/multi_cased_L-12_H-768_A-12/',
                           task=kashgari.CLASSIFICATION,
                           sequence_length=200)

(X_train, X_val, X_test, y_train, y_val, y_test, metadata) = generate_data('combined_strict.txt', pretrained=bert_embed)

bert_embed.processor.add_bos_eos = False

model = BiGRU_Model(bert_embed)

eval_callback = EvalCallBack(kash_model=model, valid_x=X_val, valid_y=y_val, step=5)
test_callback = EvalCallBack(kash_model=model, valid_x=X_test, valid_y=y_test, step=5)
early_stopping = EarlyStopping(patience=PATIENCE, monitor='val_acc')
model_checkpoint = KashgariModelCheckpoint(filepath=TARGET_DIRECTORY + 'senti_pretrained_mul1.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

model.build_model(X_train, y_train, X_val, y_val)

from keras import optimizers
optimizer = optimizers.Adam(clipnorm=1.)
print("Using gradient clipping!")
model.compile_model(optimizer=optimizer)

model.fit(X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE, epochs=EPOCH, callbacks=[eval_callback, test_callback, early_stopping]) # batch_size=100

model.evaluate(X_test, y_test)
# model.save(TARGET_DIRECTORY + 'senti_pretrained_mul.h5')
loaded_model = kashgari.utils.load_model(TARGET_DIRECTORY + 'senti_pretrained_mul1.h5')

Y_pred = loaded_model.predict(X_test)

mapping = {
    'positive': 0,
    'neutral': 1,
    'negative': 2
}

print("Y_pred:", Y_pred)
Y_pred = [mapping[pred] for pred in Y_pred]
y_test = y_test.argmax(1)

from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test, Y_pred)
print(confusion_matrix)

print(metrics.classification_report(y_test, Y_pred))
