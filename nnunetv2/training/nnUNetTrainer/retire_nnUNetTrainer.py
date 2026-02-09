"""
Retired implementation of nnUNetTrainer.run_training() method with original print statements.
Preserved for reference.
"""


def run_training_original(self):
    self.on_train_start()

    for epoch in range(self.current_epoch, self.num_epochs):
        self.on_epoch_start()

        self.on_train_epoch_start()
        train_outputs = []
        for batch_id in range(self.num_iterations_per_epoch):
            train_outputs.append(self.train_step(next(self.dataloader_train)))
        self.on_train_epoch_end(train_outputs)

        with torch.no_grad():
            self.on_validation_epoch_start()
            val_outputs = []
            for batch_id in range(self.num_val_iterations_per_epoch):
                val_outputs.append(self.validation_step(next(self.dataloader_val)))
            self.on_validation_epoch_end(val_outputs)

        self.on_epoch_end()

    self.on_train_end()
