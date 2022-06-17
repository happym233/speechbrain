import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml


class AudioMNISTBrain(sb.Brain):

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        # print(batch.signal)
        feats = self.prepare_features(batch.signal, stage)
        if 'embedding' in self.modules.keys():
            feats = self.modules.embedding(feats)
        predictions = self.modules.classifier(feats)
        predictions = torch.flatten(predictions, 1)
        return predictions

    def prepare_features(self, wavs, stage):
        wavs, lens = wavs
        feats = self.modules.compute_features(wavs)
        feats = self.modules.norm(feats, lens)
        if 'embedding' not in self.modules.keys():
            feats = torch.unsqueeze(feats, dim=3)
        return feats

    def compute_objectives(self, predictions, batch, stage):
        label_encoded = batch.label
        loss = torch.nn.CrossEntropyLoss()
        l = loss(predictions, label_encoded)
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, label_encoded)
        return l

    def on_stage_start(self, stage, epoch=None):
        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=torch.nn.CrossEntropyLoss(),
        )

        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error": self.error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:
            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["error"])

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )


def dataio_prep(hparams):
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": hparams["save_folder"]}
    )
    val_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["val_csv"],
        replacements={"data_root": hparams["save_folder"]}
    )
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"],
        replacements={"data_root": hparams["save_folder"]}
    )
    datasets = [train_data, val_data, test_data]
    label_encoder.update_from_didataset(train_data, "digit_label")

    @sb.utils.data_pipeline.takes("file_path")
    @sb.utils.data_pipeline.provides("signal")
    def audio_pipeline(file_path):
        return sb.dataio.dataio.read_audio(file_path)

    @sb.utils.data_pipeline.takes("digit_label")
    @sb.utils.data_pipeline.provides("label", "label_encoded")
    def label_to_tensor(digit_label):
        yield int(digit_label)
        label_encoded = label_encoder.encode_label_torch(digit_label)
        yield label_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    sb.dataio.dataset.add_dynamic_item(datasets, label_to_tensor)
    sb.dataio.dataset.set_output_keys(datasets, ["id", "signal", "label", "label_encoded"])

    return datasets


if __name__ == '__main__':
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    # print(hparams_file)
    from recipes.digit_recognition.audioMNIST.audioMNIST_prepare import prepare_audioMNIST

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    prepare_audioMNIST(hparams['data_folder'], hparams['save_folder'], hparams['split_ratio'],
                       reprepare=False)
    train_data, val_data, test_data = dataio_prep(hparams)
    signal = train_data[200]['signal']
    audio_MNIST_brain = AudioMNISTBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )
    audio_MNIST_brain.fit(
        epoch_counter=audio_MNIST_brain.hparams.epoch_counter,
        train_set=train_data,
        valid_set=val_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
    audio_MNIST_brain.evaluate(test_set=test_data, test_loader_kwargs=hparams["dataloader_options"])
