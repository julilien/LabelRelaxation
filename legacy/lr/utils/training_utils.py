class LearningRateScheduleProvider(object):
    def __init__(self, steps=None, init_lr=1e-3, multiplier=0.1, warmup=0):
        if steps is None:
            self.steps = [80, 120, 160, 180]
        else:
            self.steps = steps
        self.init_lr = init_lr
        self.multiplier = multiplier
        self.warmup = warmup

    def get_lr_schedule(self, epoch):
        """Learning Rate Schedule

        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.

        # Arguments
            epoch (int): The number of epochs

        # Returns
            lr (float32): learning rate
        """
        if self.warmup > 0 and epoch < self.warmup:
            return (epoch + 1) * self.init_lr / self.warmup

        lr = self.init_lr
        multiplier = self.multiplier
        for loc_steps in self.steps:
            if epoch > loc_steps:
                lr *= multiplier
            else:
                break

        return lr


def conduct_gen_model_training(model, training_gen_fn, validation_data, model_parameters, callbacks,
                               verbose=1):
    model.fit(x=training_gen_fn, steps_per_epoch=model_parameters.get_parameter("train_steps_per_epoch"),
              epochs=model_parameters.get_parameter("epochs"), callbacks=callbacks, validation_data=validation_data,
              validation_steps=model_parameters.get_parameter("validation_steps"), verbose=verbose)


def conduct_simple_model_training(model, x_train, y_train, validation_data, model_parameters, callbacks, shuffle=True,
                                  verbose=1):
    model.fit(x_train, y_train, batch_size=model_parameters.get_parameter("batch_size"),
              epochs=model_parameters.get_parameter("epochs"), validation_data=validation_data, shuffle=shuffle,
              callbacks=callbacks, verbose=verbose)
