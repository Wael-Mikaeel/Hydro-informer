# models.py

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv1D, Bidirectional, GRU, LSTM, Reshape, Concatenate, SpatialDropout1D, TimeDistributed, Flatten, Multiply, Add, LayerNormalization, Dropout, MultiHeadAttention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from utils import custom_loss_extreme_focus

def build_model_1(input_shape, output_shape, d_model=64):
    model = Sequential()
    model.add(Conv1D(filters=d_model, kernel_size=3, activation='relu', padding="same", input_shape=input_shape))
    model.add(Bidirectional(GRU(d_model, activation='relu'), backward_layer=GRU(d_model, activation='relu', go_backwards=True)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_shape[0]))
    model.add(Reshape((output_shape[0], 1)))
    return model

def compile_and_train_model_1(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=16):
    callback = EarlyStopping(monitor='loss', patience=7)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-6)
    
    model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss_extreme_focus(y_true, y_pred, threshold=8), metrics=['mse', 'mae'])
    
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[reduce_lr])
    
    loss = model.evaluate(X_test, y_test)
    return loss

def build_model_2(input_shapes, d_model=64, num_heads=2):
    input_layers = [Input(shape=(shape[1], 1)) for shape in input_shapes]
    
    embedding_layers = [Dense(d_model, activation='relu')(input) for input in input_layers]
    concat_layer = Concatenate()(embedding_layers)

    conv_layer = Conv1D(filters=d_model, kernel_size=6, activation='relu', padding="same")(concat_layer)
    conv_layer = SpatialDropout1D(0.2)(conv_layer)
    
    lstm_layer = Bidirectional(GRU(d_model, activation='relu', return_sequences=True),
                               backward_layer=LSTM(d_model, activation='relu', return_sequences=True, go_backwards=True),
                               merge_mode='sum')(conv_layer)
    
    q_layer = Dense(d_model)(lstm_layer)
    k_layer = Dense(d_model)(lstm_layer)
    v_layer = Dense(d_model)(lstm_layer)
    multi_head_attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(q_layer, k_layer, v_layer)
    
    gating_layer = Dense(d_model, activation='sigmoid')(concat_layer)
    gated_output = Multiply()([multi_head_attention, gating_layer])
    
    residual_connection = Add()([lstm_layer, gated_output])
    residual_connection = LayerNormalization(epsilon=1e-6)(residual_connection)
    residual_connection_ = Dropout(0.1)(residual_connection)
    
    lstm_layer_2 = Bidirectional(GRU(d_model, activation='relu', return_sequences=True),
                                 backward_layer=LSTM(d_model, activation='relu', return_sequences=True, go_backwards=True),
                                 merge_mode='sum')(concat_layer)
    
    conv_residual = Conv1D(filters=d_model, kernel_size=6, activation='relu', padding="same")(lstm_layer_2)
    
    multi_head_attention_2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(conv_residual, residual_connection_, residual_connection_)
    
    residual_connection_2 = Add()([conv_residual, multi_head_attention_2])
    
    output_layer = TimeDistributed(Dense(d_model*2))(residual_connection_2)
    output_layer = TimeDistributed(Dense(d_model))(output_layer)
    output_layer = Flatten()(output_layer)
    output_layer = Dense(12)(output_layer)
    output_layer = Reshape((12, 1))(output_layer)
    
    model = Model(inputs=input_layers, outputs=output_layer)
    return model

def compile_and_train_model_2(model, X_train_list, y_train, X_test_list, y_test, epochs=20, batch_size=16):
    callback = EarlyStopping(monitor='loss', patience=5)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-6)
    model_checkpoint = ModelCheckpoint(filepath='best_modell.h5', save_best_only=True, save_weights_only=True, monitor='val_mse', mode='min', verbose=1)
    
    model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss_extreme_focus(y_true, y_pred, threshold=20), metrics=['mse'])
    
    model.fit(X_train_list, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[reduce_lr, model_checkpoint])
    
    loss = model.evaluate(X_test_list, y_test)
    return loss
