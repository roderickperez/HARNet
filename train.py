import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time
import streamlit as st
import pandas as pd


def app():
    """
    Main function that contains the application to train keras based models.
    """
    @tf.function
    def train_step(x, y):
        """
        Tensorflow function to compute gradient, loss and metric defined globally
        based on given data and model.
        """
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_acc_metric.update_state(y, logits)
        return loss_value

    @tf.function
    def test_step(x, y):
        """
        Tensorflow function to compute predicted loss and metric using sent
        data from the trained model.
        """
        val_logits = model(x, training=False)
        val_acc_metric.update_state(y, val_logits)
        return loss_fn(y, val_logits)

    networkParametersExpander = st.sidebar.expander(
        "Neural Network Parameters")

    inputShape = networkParametersExpander.number_input(
        'Shape of Input', min_value=1, value=1, step=1)

    denseLayers = networkParametersExpander.number_input(
        'Dense Layers', min_value=1, value=1, step=1)

    nodesDenseLayers = networkParametersExpander.number_input(
        'Nodes in Dense Layers', min_value=1, value=1, step=1)

    activationFunction = networkParametersExpander.selectbox(
        'Activation Function', ['relu', 'sigmoid', 'tanh'])

    outputNodes = networkParametersExpander.number_input(
        'Output Nodes', min_value=1, value=1, step=1)

    optimizer = networkParametersExpander.selectbox(
        'Optimizer', ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'])

    lossFunction = networkParametersExpander.selectbox(
        'Loss Function', ['Mean Squared Error', 'Binary Crossentropy', 'Categorical Crossentropy', 'Sparse Categorical Crossentropy'])

    batchSize = networkParametersExpander.number_input(
        'Batch Size', min_value=1, value=1, step=1)

    epochs = networkParametersExpander.number_input(
        'Select Number of Epochs', min_value=1, value=10, step=1)

    saveModelCondition = networkParametersExpander.radio(
        'Would you want to save the model?', ['Yes', 'No'])

    if saveModelCondition == 'Yes':

        modelName = networkParametersExpander.text_input(
            'Model Name To Save', str(
                f"Model_InputShape_{inputShape}_DenseLayers_{denseLayers}_NodesDenseLayers_{nodesDenseLayers}_ActivationFunction_{activationFunction}_OutputNodes_{outputNodes}_Optimizer_{optimizer}_LossFunction_{lossFunction}_BatchSize_{batchSize}_Epochs_{epochs}"))

        saveCondition = networkParametersExpander.radio("Choose save condition...",
                                                        ("train acc", "val acc", "train loss", "val loss"))

    df = pd.read_csv('data/USEPUINDXD_data.csv', sep=';')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d').dt.date
    df.index = df['Date']
    df.drop('Date', axis=1, inplace=True)
    # Remove '.' values for NA
    df = df.replace('.', np.NaN)
    # Delete NA in dataframe
    df = df.dropna()

    tab1, tab2, tab3 = st.tabs(["Full Dataset", "Training", "Test"])

    with tab1:
        st.write('')
        st.write(len(df))
        st.dataframe(df)

    with tab2:
        st.write('Train')
        train_size = 0.8
        df_train = df[: int(len(df) * train_size)]
        st.write(len(df_train))
        st.dataframe(df_train)

        X_train = df_train.index
        y_train = df_train.values

        tab21, tab22 = st.tabs(["X_train", "y_train"])

        with tab21:
            st.dataframe(X_train)
        with tab22:
            st.dataframe(y_train)

    with tab3:
        st.write('Test')
        df_test = df[int(len(df) * train_size):]
        st.write(len(df_test))
        st.dataframe(df_test)

        X_test = df_test.index
        y_test = df_test.values

        tab31, tab32 = st.tabs(["X_test", "y_test"])

        with tab31:
            st.dataframe(X_test)
        with tab32:
            st.dataframe(y_test)
            
    # Calculate number of training steps.
    train_steps_per_epoch = len(X_train) //batchSize

    if networkParametersExpander.button("Train"):
        tab1, tab2, tab3 = st.tabs(["Model Summary", "Training", "Results"])
        with tab1:
            pass

        with tab2:
            st.write(
                "Starting training with {} epochs...".format(epochs))

            # epochs = 2
            for epoch in range(epochs):
                print("\nStart of epoch %d" %
                      (epoch,))
                st.write(
                    "Epoch {}".format(epoch+1))
                start_time = time.time()
                progress_bar = st.progress(0.0)
                percent_complete = 0
                epoch_time = 0
                # Creating empty placeholder to update each step result in epoch.
                st_t = st.empty()

                train_loss_list = []
                # Iterate over the batches of the dataset.
                for step, (x_batch_train, y_batch_train) in enumerate(df_train):
                    start_step = time.time()
                    loss_value = train_step(
                        x_batch_train, y_batch_train)
                    end_step = time.time()
                    epoch_time += (end_step -
                                   start_step)
                    train_loss_list.append(
                        float(loss_value))

                    # Log every 200 batches.
                    if step % 100 == 0:
                        print(
                            "Training loss (for one batch) at step %d: %.4f"
                            % (step, float(loss_value))
                        )
                        print("Seen so far: %d samples" % (
                            (step + 1) * batchSize))
                        step_acc = float(
                            train_acc_metric.result())
                        percent_complete = (
                            (step/train_steps_per_epoch))
                        progress_bar.progress(
                            percent_complete)
                        st_t.write("Duration : {0:.2f}s, Training acc. : {1:.4f}"
                                   .format((epoch_time), float(step_acc)))

                progress_bar.progress(1.0)

                # Display metrics at the end of each epoch.
                train_acc = train_acc_metric.result()
                print("Training acc over epoch: %.4f" % (
                    float(train_acc),))

                # Reset training metrics at the end of each epoch
                train_acc_metric.reset_states()

                # Find epoch training loss.
                print(train_loss_list)
                train_loss = round(
                    (sum(train_loss_list)/len(train_loss_list)), 5)

                val_loss_list = []
                # Run a validation loop at the end of each epoch.
                for x_batch_val, y_batch_val in val_dataset:
                    val_loss_list.append(
                        float(test_step(x_batch_val, y_batch_val)))

                # Find epoch validation loss.
                val_loss = round(
                    (sum(val_loss_list)/len(val_loss_list)), 5)

                val_acc = val_acc_metric.result()
                val_acc_metric.reset_states()

                print("Validation acc: %.4f" %
                      (float(val_acc),))
                print("Time taken: %.2fs" %
                      (time.time() - start_time))
                st_t.write("Duration : {0:.2f}s, Training acc. : {1:.4f}, Validation acc.:{2:.4f}"
                           .format((time.time() - start_time), float(train_acc), float(val_acc)))

                # Check if model needs to be saved, and if yess, then with what condition.
                if modelName:
                    if saveCondition:
                        if epoch == 0:
                            best_train_acc = train_acc
                            best_train_loss = train_loss
                            best_val_loss = val_loss
                            best_val_acc = val_acc

                            # Save first model.
                            model.save("./model/"+modelName+".h5", overwrite=True,
                                       include_optimizer=True)
                            if saveCondition in ("train acc", "val acc"):
                                st.write("Saved model {} as {} increased from 0 to {}."
                                         .format(modelName+".h5", saveCondition,
                                                 round(train_acc, 3) if saveCondition == "train acc" else round(val_acc, 3)))
                            else:
                                st.write("Saved model {} as {} decreased from infinite to {}."
                                         .format(modelName+".h5", saveCondition,
                                                 round(train_loss, 3) if saveCondition == "train loss" else round(val_loss, 3)))
                        else:
                            if saveCondition == "train acc":
                                if train_acc >= best_train_acc:
                                    model.save("./model/"+modelName+".h5", overwrite=True,
                                               include_optimizer=True)
                                    st.write("Saved model {} as {} increased from {} to {}."
                                             .format(modelName+".h5", saveCondition,
                                                     round(best_train_acc, 3), round(train_acc, 3)))
                                    best_train_acc = train_acc
                                else:
                                    st.write("Not saving model as {} did not increase from {}."
                                             .format(saveCondition, round(best_train_acc, 3)))
                            elif saveCondition == "val acc":
                                if val_acc >= best_val_acc:
                                    model.save("./model/"+modelName+".h5", overwrite=True,
                                               include_optimizer=True)
                                    st.write("Saved model {} as {} increased from {} to {}."
                                             .format(modelName+".h5", saveCondition,
                                                     round(best_val_acc, 3), round(val_acc, 3)))
                                    best_val_acc = val_acc
                                else:
                                    st.write("Not saving model as {} did not increase from {}."
                                             .format(saveCondition, round(best_val_acc, 3)))

                            elif saveCondition == "train loss":
                                if train_loss >= best_train_loss:
                                    model.save("./model/"+modelName+".h5", overwrite=True,
                                               include_optimizer=True)
                                    st.write("Saved model {} as {} decreased from {} to {}."
                                             .format(modelName+".h5", saveCondition,
                                                     round(best_train_loss, 3), round(train_loss, 3)))
                                    best_train_loss = train_loss
                                else:
                                    st.write("Not saving model as {} did not increase from {}."
                                             .format(saveCondition, round(best_train_loss, 3)))

                            elif saveCondition == "val loss":
                                if val_loss >= best_val_loss:
                                    model.save("./model/"+modelName+".h5", overwrite=True,
                                               include_optimizer=True)
                                    st.write("Saved model {} as {} decreased from {} to {}."
                                             .format(modelName+".h5", saveCondition,
                                                     round(best_val_loss, 3), round(val_loss, 3)))
                                    best_val_loss = val_loss
                                else:
                                    st.write("Not saving model as {} did not increase from {}."
                                             .format(saveCondition, round(best_val_loss, 3)))


if __name__ == '__main__':
    app()
