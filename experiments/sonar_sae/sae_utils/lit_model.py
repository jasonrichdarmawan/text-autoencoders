from typing import (
    Any,
    Optional,
    Iterable,
    List,
)

from jaxtyping import Float

import torch

import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from sae_lens import (
    __version__,
)

from sae_lens.training.activation_scaler import ActivationScaler

from sae_lens.training.sae_trainer import (
    _update_sae_lens_training_version,
    _log_feature_sparsity,
    _unwrap_item,
)

from sae_lens.config import (
    LanguageModelSAERunnerConfig,
)


from sae_lens.saes.sae import (
    T_TRAINING_SAE,
    TrainCoefficientConfig,
    TrainStepInput,
    TrainStepOutput,
)

from sae_lens.training.optim import (
    CoefficientScheduler,
    get_lr_scheduler,
)

import wandb

from fairseq2.data import Collater, read_sequence
from fairseq2.data.text.tokenizers import TextTokenizer

from fairseq2.nn.padding import pad_seqs
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.nn.padding import PaddingMask

from fairseq2.models.sequence import SequenceBatch

from fairseq2.generation import BeamSearchSeq2SeqGenerator

from fairseq2.generation.text import SequenceToTextConverter

from sonar.models.sonar_translation.model import (
    SonarEncoderDecoderModel,
    DummyEncoderModel,
)

from sonar.inference_pipelines.text import precision_context


class LitModel(L.LightningModule):
    """
    Partial code from sae_lens.LanguageModelSAETrainingRunner
    """

    act_freq_scores: Float[torch.Tensor, "d_sae"]
    """
    how many times each feature (neuron) has been active within 
    `self.cfg.feature_sampling_window` training steps
    """

    n_forward_passes_since_fired: Float[torch.Tensor, "d_sae"]
    """
    how many training steps have passed since each feature (neuron)
    last "fired" (i.e., produced a nonzero activation)
    """

    n_frac_active_samples: Float[torch.Tensor, "1"]
    """
    how many samples have been processed within `self.cfg.feature_sampling_window`
    training steps
    """

    def __init__(
        self,
        cfg: LanguageModelSAERunnerConfig,
        model: SonarEncoderDecoderModel,
        encoder_tokenizer: TextTokenizer,
        decoder_tokenizer: TextTokenizer,
        sae: T_TRAINING_SAE,
    ):
        """
        Partial code from: sae_lens.training.sae_trainer.SAETrainer.__init__
        """
        super().__init__()

        self.model = model
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.sae = sae

        self.cfg = cfg
        # self.cfg.weight_normalize_eps = 1e-8
        # self.cfg.resample_scale = 0.5
        # self.cfg.resample_freq = 2500
        self.activation_scaler = ActivationScaler()

        self.register_buffer(
            name="n_training_samples",
            tensor=torch.zeros(1),
        )

        _update_sae_lens_training_version(self.sae)

        self.register_buffer(
            name="act_freq_scores",
            tensor=torch.zeros(self.sae.cfg.d_sae),
        )
        self.register_buffer(
            name="n_forward_passes_since_fired",
            tensor=torch.zeros(self.sae.cfg.d_sae),
        )
        self.register_buffer(
            name="n_frac_active_samples",
            tensor=torch.zeros(1),
        )
        self.accumulated_train_step_log_dicts = []
        for name, param in self.sae.named_parameters():
            if "scaling_factor" in name:
                param.requires_grad = False

        self.coefficient_schedulers = {}
        for name, coeff_cfg in self.sae.get_coefficients().items():
            if not isinstance(coeff_cfg, TrainCoefficientConfig):
                coeff_cfg = TrainCoefficientConfig(value=coeff_cfg, warm_up_steps=0)
            self.coefficient_schedulers[name] = CoefficientScheduler(
                warm_up_steps=coeff_cfg.warm_up_steps,
                final_value=coeff_cfg.value,
            )

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.save_hyperparameters(
            logger=False,
            ignore=[
                "model",
                "encoder_tokenizer",
                "decoder_tokenizer",
                "sae",
            ],
        )

        self.strict_loading = False

    def tokenize_text(self, texts: list[str], langs: list[str]):
        """
        Partial code from sonar.inference_pipelines.text.text.TextToEmbeddingModelPipeline.predict
        """

        def encode_fn(text: str, source_lang: str) -> torch.Tensor:
            tokenizer_encoder = self.encoder_tokenizer.create_encoder(
                lang=source_lang,
                device=self.device,
            )
            return tokenizer_encoder(text)[:512]  # truncate to max length 512

        seqs = [encode_fn(text, lang) for text, lang in zip(texts, langs)]
        collater = Collater(pad_value=self.encoder_tokenizer.vocab_info.pad_idx)
        batch = collater(seqs)
        tokens, padding_mask = get_seqs_and_padding_mask(data=batch)
        return tokens, padding_mask

    def encode_text(
        self,
        seqs: Float[torch.Tensor, "batch seq_len"],
        padding_mask: Optional[PaddingMask],
    ):
        """
        Partial code from sonar.models.sonar_translation.model.SonarEncoderDecoderModel.encode
        """
        batch = SequenceBatch(seqs, padding_mask)
        sonar_output_encoder = self.model.encoder(batch)
        return sonar_output_encoder.sentence_embeddings

    def decode_embedding(
        self,
        embeddings: Float[torch.Tensor, "batch d_in"],
        target_lang: list[str],
    ):
        """
        Partial code from sonar.inference_pipelines.text.embedding_to_text.EmbeddingToTextModelPipeline.predict
        """
        dummy_encoder = DummyEncoderModel(model_dim=self.model.decoder.model_dim)
        model = SonarEncoderDecoderModel(
            encoder=dummy_encoder,
            decoder=self.model.decoder,
        ).eval()
        generator = BeamSearchSeq2SeqGenerator(model)

        def _do_translate(x: tuple[torch.Tensor, str]):
            tensor, lang = x
            converter = SequenceToTextConverter(
                generator,
                self.decoder_tokenizer,
                task="translation",
                target_lang=lang,
            )
            texts, _ = converter.batch_convert(
                torch.stack([tensor]).to(self.device),
                None,
            )
            return texts

        pipeline: Iterable = (
            read_sequence(
                list(
                    zip(
                        list(embeddings),
                        list(target_lang),
                    )
                )
            )
            .map(_do_translate)
            .and_return()
        )

        with precision_context(self.model.dtype):
            results: List[List[str]] = list(iter(pipeline))

        return [x for y in results for x in y]

    def get_target_seqs(self, texts: list[str], langs: list[str]):
        """
        Reference: https://github.com/facebookresearch/SONAR/issues/73#issuecomment-3200562749
        """
        return [
            self.decoder_tokenizer.create_encoder(
                lang=lang,
                mode="target",
                device=self.device,
            )(text)[
                :512
            ]  # truncate to max length 512
            for text, lang in zip(texts, langs)
        ]

    def get_logits(
        self,
        embeddings: Float[torch.Tensor, "batch d_in"],
        target_seqs: Float[torch.Tensor, "batch target_seq_len"],
    ):
        """
        Reference: https://github.com/facebookresearch/SONAR/issues/73#issuecomment-3200562749
        """
        # Prepare the batch for the model
        padded, mask = pad_seqs(seqs=target_seqs)

        # feed the batch to the mdoel in three steps (embeddings + decoder body + output projection)
        seqs, padding_mask = self.model.decoder.decoder_frontend(
            seqs=padded,
            padding_mask=mask,
        )
        decoder_output, decoder_padding_mask = self.model.decoder.decoder(
            seqs=seqs,
            padding_mask=mask,
            encoder_output=embeddings.unsqueeze(1),
        )
        logits = self.model.decoder.final_proj(x=decoder_output)

        return logits, padded

    def get_decoder_loss(
        self,
        logits: Float[torch.Tensor, "batch target_seq_len vocab_size"],
        padded: Float[torch.Tensor, "batch target_seq_len"],
    ):
        """
        Reference: https://github.com/facebookresearch/SONAR/issues/73#issuecomment-3200562749
        """
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

        # the "targets" are all tokens except the first one (beginning-of-sentence)
        labels = padded[:, 1:].clone()
        # make the loss ignore the padding tokens
        labels[labels == 0] = -100
        # make the loss ignore the first label (which is always the language tag; it doesn't have to be predicted)
        labels[:, 0] = -100
        loss = loss_fn(logits[:, :-1].reshape(-1, logits.size(-1)), labels.view(-1))
        per_token_loss = loss.view(logits[:, :-1].shape[:2])
        # per-sentence loss is the sum of its per-token losses
        per_sent_loss = per_token_loss.sum(-1)
        # we also compute the number of tokens, so that we could normalize the total
        # loss by the total number of tokens
        per_sent_toks = (labels != -100).sum(-1)

        return per_sent_loss, per_sent_toks

    def on_save_checkpoint(self, checkpoint):
        """
        Partial code from sae_lens.training.sae_trainer.SAETrainer.fit
        """
        # fold the estimated norm scaling into the weights
        if self.activation_scaler.scaling_factor is not None:
            scaling_factor = self.activation_scaler.scaling_factor
            checkpoint["state_dict"]["sae.W_enc"] *= scaling_factor
            checkpoint["state_dict"]["sae.b_dec"] /= scaling_factor
            checkpoint["state_dict"]["sae.W_dec"] /= scaling_factor

        checkpoint["state_dict"] = {
            k: v for k, v in self.state_dict().items() if not k.startswith("model.")
        }

    def on_load_checkpoint(self, checkpoint):
        sae_state_dict = {
            k.replace("sae.", ""): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("sae.")
        }
        self.sae.load_state_dict(sae_state_dict, strict=True)
        self.n_training_samples = checkpoint["state_dict"]["n_training_samples"]
        self.act_freq_scores = checkpoint["state_dict"]["act_freq_scores"]
        self.n_forward_passes_since_fired = checkpoint["state_dict"][
            "n_forward_passes_since_fired"
        ]
        self.n_frac_active_samples = checkpoint["state_dict"]["n_frac_active_samples"]

    def configure_optimizers(self):
        """
        Partial code from sae_lens.training.sae_trainer.SAETrainer.configure_optimizers
        """
        optimizer = torch.optim.AdamW(
            params=self.sae.parameters(),
            lr=self.cfg.lr,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
        )

        accumulate = getattr(self.trainer, "accumulate_grad_batches", 1)
        scheduler = {
            "scheduler": get_lr_scheduler(
                scheduler_name=self.cfg.lr_scheduler_name,
                lr=self.cfg.lr,
                optimizer=optimizer,
                warm_up_steps=self.cfg.lr_warm_up_steps // accumulate,
                decay_steps=self.cfg.lr_decay_steps // accumulate,
                training_steps=self.cfg.total_training_steps // accumulate,
                lr_end=self.cfg.lr_end,
                num_cycles=self.cfg.n_restart_cycles,
            ),
            "interval": "step",
        }

        return [optimizer], [scheduler]

    def on_fit_start(self):
        if self.sae.cfg.normalize_activations == "expected_average_only_in":
            acts_iter = iter(self.train_embedding_iterator())
            self.activation_scaler.estimate_scaling_factor(
                d_in=self.sae.cfg.d_in,
                data_provider=acts_iter,
                n_batches_for_norm_estimate=(
                    int(1e3) * self.trainer.accumulate_grad_batches
                ),
            )
            rank_zero_only(
                print(
                    f"Estimated activation scaling factor: {self.activation_scaler.scaling_factor}"
                )
            )

    def train_embedding_iterator(self):
        # device = self.sae.device
        # batch_list = []
        # for batch in self.trainer.datamodule.train_dataloader():
        #     batch_list.append(batch["embedding1"].to(device))
        #     if len(batch_list) == self.trainer.accumulate_grad_batches:
        #         yield torch.cat(batch_list, dim=0)
        #         batch_list = []
        for batch in self.trainer.datamodule.train_dataloader():
            yield torch.cat(
                [
                    batch[0]["nllb_200_6m_sample_embedding"]["embedding1"].to(
                        self.sae.device
                    ),
                    batch[0]["nllb_primary_datasets_embedding"]["embedding1"].to(
                        self.sae.device
                    ),
                ],
                dim=0,
            )

    def training_step(
        self,
        batch,
    ):
        """
        Partial code from sae_lens.training.sae_trainer.SAETrainer._train_step

        batch: dict with keys: see sae_utils.nllb_data_module.NLLBDataModule.train_dataloader
        """
        batch_size = (
            batch[0]["nllb_200_6m_sample_embedding"]["embedding1"].shape[0]
            + batch[0]["nllb_primary_datasets_embedding"]["embedding1"].shape[0]
        )
        # batch_size = batch["embedding1"].shape[0]
        self.n_training_samples += batch_size

        scaled_batch = self.activation_scaler(
            acts=torch.concat(
                [
                    batch[0]["nllb_200_6m_sample_embedding"]["embedding1"],
                    batch[0]["nllb_primary_datasets_embedding"]["embedding1"],
                ],
                dim=0,
            ),
            # acts=batch["embedding1"],
        )
        step_input = TrainStepInput(
            sae_in=scaled_batch,
            coefficients=self.get_coefficients(),
            dead_neuron_mask=self.dead_neurons,
        )
        train_step_output = self.sae.training_forward_pass(
            step_input=step_input,
        )

        with torch.no_grad():
            did_fire = (train_step_output.feature_acts > 0).float().sum(dim=-2) > 0
            self.n_forward_passes_since_fired += 1
            self.n_forward_passes_since_fired[did_fire] = 0

            # personal note:
            # suppose we take the mean over batch
            # size instead of sum, then we don't have to use
            # a different learning rate for different batch size
            # however, the original code uses sum, so we keep it that way
            self.act_freq_scores += (
                (train_step_output.feature_acts.abs() > 0).float().sum(dim=0)
            )
            self.n_frac_active_samples += batch_size

            if (self.global_step + 1) % self.cfg.logger.wandb_log_frequency == 0:
                train_step_log_dict = self._build_train_step_log_dict(
                    output=train_step_output,
                    n_training_samples=self.n_training_samples,
                )
                self.accumulated_train_step_log_dicts.append(train_step_log_dict)

        return train_step_output.loss

    def on_before_optimizer_step(self, optimizer):
        """
        Partial code from sae_lens.training.sae_trainer.SAETrainer._train_step
        """
        if (self.global_step + 1) % self.cfg.feature_sampling_window == 0:
            if self.cfg.logger.log_to_wandb:
                sparsity_log_dict = self._build_sparsity_log_dict()
                self.logger.experiment.log(
                    sparsity_log_dict,
                    step=self.global_step,
                )
            self._reset_runing_sparsity_stats()

        if self.cfg.logger.log_to_wandb:
            if (self.global_step + 1) % self.cfg.logger.wandb_log_frequency == 0:
                mean_train_step_log_dict = {
                    k: (
                        sum(d[k] for d in self.accumulated_train_step_log_dicts)
                        / len(self.accumulated_train_step_log_dicts)
                    )
                    for k in self.accumulated_train_step_log_dicts[0].keys()
                }

                self.log_dict(
                    {
                        k: v
                        for k, v in mean_train_step_log_dict.items()
                        if not k == "metrics/l0"
                    }
                )
                self.log(
                    name="metrics/l0",
                    value=mean_train_step_log_dict["metrics/l0"],
                    prog_bar=True,
                )
            self.accumulated_train_step_log_dicts.clear()

        for scheduler in self.coefficient_schedulers.values():
            scheduler.step()

        # # Resample dead neurons
        # if (self.global_step + 1) % self.cfg.resample_freq == 0:
        #     self.resample_neurons(self.last_batch)

    down_stream_reconstruction_metrics_dict = {
        "ce_loss_with_sae": [],
        "ce_loss_without_sae": [],
        "ce_loss_with_ablation": [],
    }
    sparsity_and_variance_metrics_dict = {
        "l2_norm_in": [],
        "l2_norm_out": [],
        "l2_ratio": [],
        "relative_reconstruction_bias": [],
        "l0": [],
        "l1": [],
        "explained_variance_legacy": [],
        "mse": [],
        "cossim": [],
    }
    mean_sum_of_squares = []  # for explained variance
    mean_act_per_dimension = []  # for explained variance
    mean_sum_of_resid_squared = []  # for explained variance

    def validation_step(self, batch, batch_idx):
        """
        Partial code from sae_lens.evals.get_downstream_reconstruction_metrics
        """

        embeddings = torch.concat(
            [
                batch[0]["nllb_200_6m_sample_embedding"]["embedding1"],
                batch[0]["nllb_primary_datasets_embedding"]["embedding1"],
            ],
            dim=0,
        )

        # compute_ce_loss
        self.get_downstream_reconstruction_metrics_step(
            texts=(
                batch[0]["nllb_200_6m_sample_embedding"]["text1"]
                + batch[0]["nllb_primary_datasets_embedding"]["text1"]
            ),
            langs=(
                batch[0]["nllb_200_6m_sample_embedding"]["lang1"]
                + batch[0]["nllb_primary_datasets_embedding"]["lang1"]
            ),
            embeddings=embeddings,
        )

        self.get_sparsity_and_variance_metrics_step(
            embeddings=embeddings,
            compute_l2_norms=True,
            compute_sparsity_metrics=True,
            compute_variance_metrics=True,
            compute_featurewise_density_statistics=False,
        )

    def on_validation_epoch_end(self):
        all_metrics = {
            "model_performance_preservation": {},
            "reconstruction_quality": {},
            "shrinkage": {},
            "sparsity": {},
        }

        reconstruction_metrics = self.get_downstream_reconstruction_metrics(
            compute_ce_loss=True
        )
        self.down_stream_reconstruction_metrics_dict = {
            key: [] for key in self.down_stream_reconstruction_metrics_dict.keys()
        }
        all_metrics["model_performance_preservation"].update(
            {
                # "ce_loss_score": reconstruction_metrics["ce_loss_score"],
                "ce_loss_with_sae": reconstruction_metrics["ce_loss_with_sae"],
                "ce_loss_without_sae": reconstruction_metrics["ce_loss_without_sae"],
                "ce_loss_with_ablation": reconstruction_metrics[
                    "ce_loss_with_ablation"
                ],
            }
        )

        # For ModelCheckpoint(monitor)
        self.log(
            name="model_performance_preservation.ce_loss_score",
            value=reconstruction_metrics["ce_loss_score"],
            prog_bar=True,
        )

        # Aggregate scalar metrics
        sparsity_variance_metrics = self.get_sparsity_and_variance_metrics(
            compute_variance_metrics=True,
        )
        self.sparsity_and_variance_metrics_dict = {
            key: [] for key in self.sparsity_and_variance_metrics_dict.keys()
        }
        all_metrics["shrinkage"].update(
            {
                "l2_norm_in": sparsity_variance_metrics["l2_norm_in"],
                "l2_norm_out": sparsity_variance_metrics["l2_norm_out"],
                "l2_ratio": sparsity_variance_metrics["l2_ratio"],
                "relative_reconstruction_bias": sparsity_variance_metrics[
                    "relative_reconstruction_bias"
                ],
            }
        )
        all_metrics["sparsity"].update(
            {
                "l0": sparsity_variance_metrics["l0"],
                "l1": sparsity_variance_metrics["l1"],
            }
        )
        all_metrics["reconstruction_quality"].update(
            {
                "explained_variance": sparsity_variance_metrics["explained_variance"],
                "explained_variance_legacy": sparsity_variance_metrics[
                    "explained_variance_legacy"
                ],
                "mse": sparsity_variance_metrics["mse"],
                "cossim": sparsity_variance_metrics["cossim"],
            }
        )

        for key, value in self.sae.log_histograms().items():
            all_metrics[key] = wandb.Histogram(value)

        self.logger.experiment.log(
            all_metrics,
            step=self.global_step,
        )

    @torch.no_grad()
    def get_downstream_reconstruction_metrics_step(
        self,
        texts: list[str],
        langs: list[str],
        embeddings: Float[torch.Tensor, "batch d_in"],
    ):
        target_seqs = self.get_target_seqs(
            texts=texts,
            langs=langs,
        )
        for metric_name, metric_value in self.get_recons_loss(
            embeddings=embeddings,
            target_seqs=target_seqs,
            compute_ce_loss=True,
        ).items():
            self.down_stream_reconstruction_metrics_dict[metric_name].append(
                metric_value
            )

    @torch.no_grad()
    def get_downstream_reconstruction_metrics(self, compute_ce_loss: bool):
        """
        Partial code from sae_lens.evals.get_downstream_reconstruction_metrics
        """
        metrics: dict[str, float] = {}
        for (
            metric_name,
            metric_values,
        ) in self.down_stream_reconstruction_metrics_dict.items():
            metrics[metric_name] = torch.cat(metric_values).mean().item()

        if compute_ce_loss:
            metrics["ce_loss_score"] = (
                metrics["ce_loss_with_ablation"] - metrics["ce_loss_with_sae"]
            ) / (metrics["ce_loss_with_ablation"] - metrics["ce_loss_without_sae"])

        return metrics

    @torch.no_grad()
    def get_sparsity_and_variance_metrics_step(
        self,
        embeddings: Float[torch.Tensor, "batch d_in"],
        compute_l2_norms: bool,
        compute_sparsity_metrics: bool,
        compute_variance_metrics: bool,
        compute_featurewise_density_statistics: bool,
    ):
        """
        Partial code from sae_lens.evals.get_sparsity_and_variance_metrics
        """
        original_act = embeddings
        # normalise if necessary (necessary in training only, otherwise we should fold the scaling in)
        original_act_scaled = self.activation_scaler.scale(original_act)

        # send the (maybe normalised) activations into the SAE
        sae_feature_activations = self.sae.encode(x=original_act_scaled)
        sae_out_scaled = self.sae.decode(feature_acts=sae_feature_activations)
        sae_out = self.activation_scaler.unscale(sae_out_scaled)

        masked_sae_feature_activations = sae_feature_activations
        flattened_sae_input = original_act
        flattened_sae_feature_acts = sae_feature_activations
        flattened_sae_out = sae_out

        if compute_l2_norms:
            l2_norm_in = torch.norm(flattened_sae_input, dim=-1)
            l2_norm_out = torch.norm(flattened_sae_out, dim=-1)
            l2_norm_in_for_div = l2_norm_in.clone()
            l2_norm_in_for_div[l2_norm_in_for_div < 0.0001] = 1
            l2_norm_ratio = l2_norm_out / l2_norm_in_for_div

            # Equation 10 from https://arxiv.org/abs/2404.16014
            # https://github.com/saprmarks/dictionary_learning/blob/main/evaluation.py
            x_hat_norm_squared = torch.norm(flattened_sae_out, dim=-1) ** 2
            x_dot_x_hat = (flattened_sae_input * flattened_sae_out).sum(dim=-1)
            # if relative reconstruction bias is much greater than 1,
            # it means the reconstructed vectors (x_hat) have much larger
            # norm than the input (x)
            # if x_hat is not well aligned with x (i.e. points in the different direction),
            # then the dot product x . x_hat will be much smaller than ||x_hat||^2
            # for example, if x_hat is nearly orthogonal to x, x . x_hat can be close to zero,
            # making the ratio very large
            # if x_hat points in the opposite direction, x . x_hat can even be negative
            relative_reconstruction_bias = (
                x_hat_norm_squared.mean() / x_dot_x_hat.mean()
            ).unsqueeze(0)

            self.sparsity_and_variance_metrics_dict["l2_norm_in"].append(l2_norm_in)
            self.sparsity_and_variance_metrics_dict["l2_norm_out"].append(l2_norm_out)
            self.sparsity_and_variance_metrics_dict["l2_ratio"].append(l2_norm_ratio)
            self.sparsity_and_variance_metrics_dict[
                "relative_reconstruction_bias"
            ].append(relative_reconstruction_bias)
        if compute_sparsity_metrics:
            l0 = (flattened_sae_feature_acts > 0).sum(dim=-1).float()
            l1 = flattened_sae_feature_acts.sum(dim=-1)
            self.sparsity_and_variance_metrics_dict["l0"].append(l0)
            self.sparsity_and_variance_metrics_dict["l1"].append(l1)

        if compute_variance_metrics:
            resid_sum_of_squares = (
                (flattened_sae_input - flattened_sae_out).pow(2).sum(dim=-1)
            )

            mse = resid_sum_of_squares
            # Explained variance (old, incorrect, formula)
            batched_variance_sum = (
                (flattened_sae_input - flattened_sae_input.mean(dim=0))
                .pow(2)
                .sum(dim=-1)
            )
            explained_variance_legacy = 1 - resid_sum_of_squares / batched_variance_sum
            self.sparsity_and_variance_metrics_dict["explained_variance_legacy"].append(
                explained_variance_legacy
            )
            # Individual sums for the new (correct) formula. We're taking the mean over the batch
            # dimension here to save memory, but we could also pass the full tensors and take the
            # mean later (like we do for other metrics).
            self.mean_sum_of_squares.append(
                (flattened_sae_input).pow(2).sum(dim=-1).mean(dim=0)  # scalar
            )
            self.mean_act_per_dimension.append(
                (flattened_sae_input).pow(2).mean(dim=0)  # [d_model]
            )
            self.mean_sum_of_resid_squared.append(
                resid_sum_of_squares.mean(dim=0)
            )  # scalar
            x_normed = flattened_sae_input / torch.norm(
                flattened_sae_input, dim=-1, keepdim=True
            )
            x_hat_normed = flattened_sae_out / torch.norm(
                flattened_sae_out, dim=-1, keepdim=True
            )
            cossim = (x_normed * x_hat_normed).sum(dim=-1)

            self.sparsity_and_variance_metrics_dict["mse"].append(mse)
            self.sparsity_and_variance_metrics_dict["cossim"].append(cossim)

        if compute_featurewise_density_statistics:
            sae_feature_activations_bool = (masked_sae_feature_activations > 0).float()
            total_feature_acts += sae_feature_activations_bool.sum(dim=1).sum(dim=0)
            total_feature_prompts += (sae_feature_activations_bool.sum(dim=1) > 0).sum(
                dim=0
            )

    @torch.no_grad()
    def get_sparsity_and_variance_metrics(
        self,
        compute_variance_metrics: bool,
    ):
        """
        Partial code from sae_lens.evals.get_sparsity_and_variance_metrics
        """
        metrics: dict[str, float] = {}
        for (
            metric_name,
            metric_value,
        ) in self.sparsity_and_variance_metrics_dict.items():
            metrics[metric_name] = torch.cat(metric_value).mean().item()

        if compute_variance_metrics:
            mean_sum_of_squares = torch.stack(self.mean_sum_of_squares).mean(dim=0)
            mean_act_per_dimension = torch.cat(self.mean_act_per_dimension).mean(dim=0)
            total_variance = mean_sum_of_squares - mean_act_per_dimension**2
            residual_variance = torch.stack(self.mean_sum_of_resid_squared).mean(dim=0)
            metrics["explained_variance"] = (
                1 - residual_variance / total_variance
            ).item()

        return metrics

    @torch.no_grad()
    def get_recons_loss(
        self,
        embeddings: Float[torch.Tensor, "batch d_in"],
        target_seqs: list[Float[torch.Tensor, "target_seq_len"]],
        compute_ce_loss: bool,
    ) -> dict[str, Any]:
        """
        Partial code from sae_lens.evals.get_recons_loss
        """
        metrics = {}

        original_logits, original_padded = self.get_logits(
            embeddings=embeddings,
            target_seqs=target_seqs,
        )

        original_ce_losses, original_n_toks = self.get_decoder_loss(
            logits=original_logits,
            padded=original_padded,
        )

        metrics = {}

        activations = self.activation_scaler.scale(embeddings)
        activations = self.sae.decode(self.sae.encode(activations))
        activations = self.activation_scaler.unscale(activations)
        recons_logits, recons_padded = self.get_logits(
            embeddings=activations,
            target_seqs=target_seqs,
        )
        recons_ce_losses, recons_n_toks = self.get_decoder_loss(
            logits=recons_logits,
            padded=recons_padded,
        )

        zero_abl_activations = torch.zeros_like(embeddings)
        zero_abl_logits, zero_abl_padded = self.get_logits(
            embeddings=zero_abl_activations,
            target_seqs=target_seqs,
        )
        zero_abl_ce_losses, zero_abl_n_toks = self.get_decoder_loss(
            logits=zero_abl_logits,
            padded=zero_abl_padded,
        )

        if compute_ce_loss:
            metrics["ce_loss_without_sae"] = original_ce_losses
            metrics["ce_loss_with_sae"] = recons_ce_losses
            metrics["ce_loss_with_ablation"] = zero_abl_ce_losses

        return metrics

    # @torch.no_grad()
    # def resample_neurons(self, sae_in: Float[torch.Tensor, "batch d_in"]):
    #     """
    #     Resample neurons for GatedSAE

    #     Partial code from https://arena-chapter1-transformer-interp.streamlit.app/[1.3.2]_Interpretability_with_SAEs#exercise-implement-resample-advanced
    #     """
    #     l2_loss = (
    #         (sae_in - self.sae(x=sae_in)).pow(2).mean(dim=-1)
    #     )  # shape: (batch_size)

    #     dead_latents = self.dead_neurons
    #     n_dead = dead_latents.sum()
    #     if n_dead == 0:
    #         return

    #     if l2_loss.max() < 1e-6:
    #         return

    #     distn = Categorical(probs=l2_loss.pow(2) / l2_loss.pow(2).sum())
    #     replacement_indices = distn.sample((n_dead,))

    #     replacement_values = (sae_in - self.sae.b_dec)[replacement_indices]
    #     replacement_values_normalized = replacement_values / (
    #         replacement_values.norm(dim=-1, keepdim=True)
    #         + self.cfg.weight_normalize_eps
    #     )

    #     W_enc_norm_alive_mean = (
    #         self.sae.W_enc[:, ~dead_latents].norm(dim=0).mean().item()
    #         if [~dead_latents].any()
    #         else 1.0
    #     )

    #     # New names for weights & biases to resample
    #     self.sae.W_dec.data[dead_latents, :] = replacement_values_normalized
    #     self.sae.W_enc.data[:, dead_latents] = (
    #         replacement_values_normalized.T
    #         * W_enc_norm_alive_mean
    #         * self.cfg.resample_scale
    #     )
    #     self.sae.b_mag.data[dead_latents] = 0.0
    #     self.sae.b_gate.data[dead_latents] = 0.0
    #     self.r_mag.data[dead_latents] = 0.0

    @property
    def feature_sparsity(self) -> Float[torch.Tensor, "d_sae"]:
        """
        Code from sae_lens.training.sae_trainer.SAETrainer.feature_sparsity

        the average activation frequency of each feature (neuron) in the sparse autoencoder
        over the last `self.cfg.feature_sampling_window` training steps
        """
        return self.act_freq_scores / self.n_frac_active_samples

    def _reset_runing_sparsity_stats(self):
        """
        Code from sae_lens.training.sae_trainer.SAETrainer._reset_runing_sparsity_stats
        """
        self.act_freq_scores.zero_()
        self.n_frac_active_samples.zero_()

    @property
    def dead_neurons(self) -> Float[torch.Tensor, "d_sae"]:
        """
        Code from sae_lens.training.sae_trainer.SAETrainer.dead_neurons
        """
        return (self.n_forward_passes_since_fired > self.cfg.dead_feature_window).bool()

    def get_coefficients(self):
        """
        Code from sae_lens.training.sae_trainer.SAETrainer.get_coefficients
        """
        return {
            name: scheduler.value
            for name, scheduler in self.coefficient_schedulers.items()
        }

    @torch.no_grad()
    def _build_sparsity_log_dict(self) -> dict[str, Any]:
        """
        Code from sae_lens.training.sae_trainer.SAETrainer._build_sparsity_log_dict
        """
        log_feature_sparsity = _log_feature_sparsity(
            feature_sparsity=self.feature_sparsity,
        )
        wandb_histogram = wandb.Histogram(log_feature_sparsity.cpu().numpy())
        return {
            "metrics/mean_log10_feature_sparsity": log_feature_sparsity.mean().item(),
            # on a log scale, how often features are active
            # see the [Interpreting the latent density histogram](https://arena-chapter1-transformer-interp.streamlit.app/[1.3.2]_Interpretability_with_SAEs#interpreting-the-latent-density-histogram)
            "plots/feature_density_line_chart": wandb_histogram,
            "sparsity/below_1e-5": (self.feature_sparsity < 1e-5).sum().item(),
            "sparsity/below_1e-6": (self.feature_sparsity < 1e-6).sum().item(),
            # counts how many features (neurons) have an average activation
            # frequency less than 1e-6
        }

    @torch.no_grad()
    def _build_train_step_log_dict(
        self,
        output: TrainStepOutput,
        n_training_samples: int,
    ) -> dict[str, float]:
        """
        Code from sae_lens.training.sae_trainer.SAETrainer._build_train_step_log_dict
        """
        sae_in = output.sae_in
        sae_out = output.sae_out
        feature_acts = output.feature_acts
        loss = output.loss.item()

        # metrics for currents acts
        l0 = (feature_acts > 0).float().sum(dim=-1).mean(dim=0)
        current_learning_rate = self.trainer.optimizers[0].param_groups[0]["lr"]

        per_token_l2_loss = (
            (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
        )  # shape: (batch_size,)
        # sum of squared differences (L2 loss) for each token/sample in the batch

        total_variance = (
            (sae_in - sae_in.mean(dim=0)).pow(2).sum(dim=-1)
        )  # shape: (batch_size,)
        # total variance of each input sample in the batch, relative to the mean
        # input across the batch

        explained_variance_legacy = (
            1 - per_token_l2_loss / total_variance
        )  # shape: (batch_size,)
        # shows how well the autoencoder reconstructs each individual sample
        # legacy: not averaged

        explained_variance = 1 - (
            per_token_l2_loss.mean(dim=0) / total_variance.mean(dim=0)
        )
        # averaged across the batch

        log_dict = {
            # losses
            "losses/overall_loss": loss,
            # variance explained
            "metrics/explained_variance_legacy": explained_variance_legacy.mean(
                dim=0
            ).item(),
            "metrics/explained_variance_legacy_std": explained_variance_legacy.std(
                dim=0
            ).item(),
            # standard deviation of the per-sample explained variance
            # high value: The model reconstructs some samples much better than
            # others (high variability)
            "metrics/explained_variance": explained_variance.item(),
            "metrics/l0": l0.item(),
            # how many features (neurons) are active on average
            # sparsity
            "sparsity/mean_passes_since_fired": self.n_forward_passes_since_fired.mean(
                dim=0
            ).item(),
            # how many training steps have passed since each feature (neuron)
            # last "fired" (i.e., produced a nonzero activation)
            "sparsity/dead_features": self.dead_neurons.sum(dim=0).item(),
            "details/current_learning_rate": current_learning_rate,
            "details/n_training_samples": n_training_samples,
            **{
                f"details/{name}_coefficient": scheduler.value
                for name, scheduler in self.coefficient_schedulers.items()
            },
        }
        # print(log_dict["sparsity/mean_passes_since_fired"])
        for loss_name, loss_value in output.losses.items():
            log_dict[f"losses/{loss_name}"] = _unwrap_item(loss_value)
            # mse_loss: total squared error per sample averaged over the batch

        for metric_name, metric_value in output.metrics.items():
            log_dict[f"metrics/{metric_name}"] = _unwrap_item(metric_value)

        return log_dict

    # def _run_and_log_evals(self):
    #     """
    #     Partial code from sae_lens.training.sae_trainer.SAETrainer._run_and_log_evals
    #     """
    #     if (self.global_step + 1) % (
    #         self.cfg.logger.wandb_log_frequency
    #         * self.cfg.logger.eval_every_n_wandb_logs
    #     ):
    #         self.sae.eval()
    #         eval_metrics = self.evaluator()
    #         for key, value in self.sae.log_histograms().items():
    #             eval_metrics[key] = wandb.Histogram(value)

    #         self.logger.experiment.log(
    #             eval_metrics,
    #             step=self.global_step,
    #         )
    #         self.sae.train()

    # def evaluator(self):
    #     """
    #     Partial code from sae_lens.llm_sae_training_runner.LLMSaeEvaluator
    #     """

    #     eval_config = EvalConfig(
    #         n_eval_reconstruction_batches=self.cfg.n_eval_batches,
    #         n_eval_sparsity_variance_batches=self.cfg.n_eval_batches,
    #         compute_ce_loss=True,
    #         compute_l2_norms=True,
    #         compute_sparsity_metrics=True,
    #         compute_variance_metrics=True,
    #     )

    #     eval_metrics, _ = run_evals(
    #         model=self,
    #         iterable=iter(self.trainer.datamodule.train_dataloader()),
    #         eval_config=eval_config,
    #     )  # not calculating featurwise metrics here.

    #     # Remove eval metrics that are already logged during training
    #     eval_metrics.pop("metrics/explained_variance", None)
    #     eval_metrics.pop("metrics/explained_variance_std", None)
    #     eval_metrics.pop("metrics/l0", None)
    #     eval_metrics.pop("metrics/l1", None)
    #     eval_metrics.pop("metrics/mse", None)

    #     # Remove metrics that are not useful for wandb logging
    #     eval_metrics.pop("metrics/total_tokens_evaluated", None)

    #     return eval_metrics


# def run_evals(
#     model: LitModel,
#     iterable: Iterator[dict[str, Any]],
#     eval_config: EvalConfig = EvalConfig(),
#     verbose: bool = False,
# ):
#     """
#     Partial code from sae_lens.evals.run_evals
#     """

#     all_metrics = {
#         "model_behavior_preservation": {},
#         "model_performance_preservation": {},
#         "reconstruction_quality": {},
#         "shrinkage": {},
#         "sparsity": {},
#         "token_stats": {},
#     }

#     if eval_config.compute_kl or eval_config.compute_ce_loss:
#         if eval_config.n_eval_reconstruction_batches <= 0:
#             raise ValueError(
#                 "eval_config.n_eval_reconstruction_batches must be > 0 when "
#                 "compute_kl or compute_ce_loss is True."
#             )
#         reconstruction_metrics = get_downstream_reconstruction_metrics(
#             model=model,
#             iterable=iterable,
#             compute_kl=eval_config.compute_kl,
#             compute_ce_loss=eval_config.compute_ce_loss,
#             n_batches=eval_config.n_eval_reconstruction_batches,
#             verbose=verbose,
#         )

#         if eval_config.compute_kl:
#             all_metrics["model_behavior_preservation"].update(
#                 {
#                     "kl_div_score": reconstruction_metrics["kl_div_score"],
#                     "kl_div_with_sae": reconstruction_metrics["kl_div_with_sae"],
#                 }
#             )

#         if eval_config.compute_ce_loss:
#             all_metrics["model_performance_preservation"].update(
#                 {
#                     "ce_loss_with_sae": reconstruction_metrics["ce_loss_with_sae"],
#                     "ce_loss_without_sae": reconstruction_metrics[
#                         "ce_loss_without_sae"
#                     ],
#                 }
#             )

#     if (
#         eval_config.compute_l2_norms
#         or eval_config.compute_sparsity_metrics
#         or eval_config.compute_variance_metrics
#     ):
#         if eval_config.n_eval_sparsity_variance_batches <= 0:
#             raise ValueError(
#                 "eval_config.n_eval_sparsity_variance_batches must be > 0 when "
#                 "compute_l2_norms, compute_sparsity_metrics, or compute_variance_metrics is True."
#             )
#         sparsity_variance_metrics, feature_metrics = get_sparsity_and_variance_metrics(
#             model=model,
#             iterable=iterable,
#             compute_l2_norms=eval_config.compute_l2_norms,
#             compute_sparsity_metrics=eval_config.compute_sparsity_metrics,
#             compute_variance_metrics=eval_config.compute_variance_metrics,
#             compute_featurewise_density_statistics=eval_config.compute_featurewise_density_statistics,
#             n_batches=eval_config.n_eval_sparsity_variance_batches,
#             verbose=verbose,
#         )

#         if eval_config.compute_l2_norms:
#             all_metrics["shrinkage"].update(
#                 {
#                     "l2_norm_in": sparsity_variance_metrics["l2_norm_in"],
#                     "l2_norm_out": sparsity_variance_metrics["l2_norm_out"],
#                     "l2_ratio": sparsity_variance_metrics["l2_ratio"],
#                     "relative_reconstruction_bias": sparsity_variance_metrics[
#                         "relative_reconstruction_bias"
#                     ],
#                 }
#             )

#         if eval_config.compute_sparsity_metrics:
#             all_metrics["sparsity"].update(
#                 {
#                     "l0": sparsity_variance_metrics["l0"],
#                     "l1": sparsity_variance_metrics["l1"],
#                 }
#             )

#         if eval_config.compute_variance_metrics:
#             all_metrics["reconstruction_quality"].update(
#                 {
#                     "explained_variance": sparsity_variance_metrics[
#                         "explained_variance"
#                     ],
#                     "explained_variance_legacy": sparsity_variance_metrics[
#                         "explained_variance_legacy"
#                     ],
#                     "mse": sparsity_variance_metrics["mse"],
#                     "cossim": sparsity_variance_metrics["cossim"],
#                 }
#             )
#     else:
#         feature_metrics = {}

#     if eval_config.compute_featurewise_weight_based_metrics:
#         feature_metrics |= get_featurewise_weight_based_metrics(sae=model.sae)

#     if len(all_metrics) == 0:
#         raise ValueError(
#             "No metrics were computed, please set at least one metric to True."
#         )

#     # Remove empty metric groups
#     all_metrics = {k: v for k, v in all_metrics.items() if v}

#     return all_metrics, feature_metrics


# def get_downstream_reconstruction_metrics(
#     model: LitModel,
#     iterable: Iterator[dict[str, Any]],
#     compute_kl: bool,
#     compute_ce_loss: bool,
#     n_batches: int,
#     verbose: bool = False,
# ):
#     """
#     Partial code from sae_lens.evals.get_downstream_reconstruction_metrics
#     """
#     metrics_dict = {}
#     if compute_kl:
#         metrics_dict["kl_div_with_sae"] = []
#     if compute_ce_loss:
#         metrics_dict["ce_loss_with_sae"] = []
#         metrics_dict["ce_loss_without_sae"] = []

#     batch_iter = range(n_batches)
#     if verbose:
#         batch_iter = tqdm(batch_iter, desc="Reconstruction Batches")

#     for _ in batch_iter:
#         batch = next(iterable)
#         target_seqs = model.get_target_seqs(
#             texts=batch["text1"],
#             langs=batch["lang1"],
#         )
#         for metric_name, metric_value in get_recons_loss(
#             model=model,
#             embeddings=batch["embedding1"],
#             target_seqs=target_seqs,
#             compute_kl=compute_kl,
#             compute_ce_loss=compute_ce_loss,
#         ).items():
#             metrics_dict[metric_name].append(metric_value)

#     metrics: dict[str, float] = {}
#     for metric_name, metric_values in metrics_dict.items():
#         metrics[f"{metric_name}"] = torch.cat(metric_values).mean().item()

#     return metrics


# @torch.no_grad()
# def get_recons_loss(
#     model: LitModel,
#     embeddings: Float[torch.Tensor, "batch d_in"],
#     target_seqs: list[Float[torch.Tensor, "target_seq_len"]],
#     compute_ce_loss: bool,
#     compute_kl: bool,
# ) -> dict[str, Any]:
#     """
#     Partial code from sae_lens.evals.get_recons_loss
#     """
#     metrics = {}

#     original_logits, original_padded = model.get_logits(
#         embeddings=embeddings,
#         target_seqs=target_seqs,
#     )

#     original_losses, original_n_toks = model.get_decoder_loss(
#         logits=original_logits,
#         padded=original_padded,
#     )
#     original_ce_loss = original_losses.sum() / original_n_toks.sum()

#     metrics = {}

#     activations = model.activation_scaler.scale(embeddings)
#     activations = model.sae.decode(model.sae.encode(activations))
#     activations = model.activation_scaler.unscale(activations)
#     recons_logits, recons_padded = model.get_logits(
#         embeddings=activations,
#         target_seqs=target_seqs,
#     )
#     recons_losses, recons_n_toks = model.get_decoder_loss(
#         logits=recons_logits,
#         padded=recons_padded,
#     )
#     recons_ce_loss = recons_losses.sum() / recons_n_toks.sum()

#     if compute_kl:
#         recons_kl_div = _kl(original_logits, recons_logits)
#         metrics["kl_div_with_sae"] = recons_kl_div

#     if compute_ce_loss:
#         metrics["ce_loss_with_sae"] = recons_ce_loss
#         metrics["ce_loss_without_sae"] = original_ce_loss

#     return metrics


# def get_sparsity_and_variance_metrics(
#     model: LitModel,
#     iterable: Iterator[dict[str, Any]],
#     n_batches: int,
#     compute_l2_norms: bool,
#     compute_sparsity_metrics: bool,
#     compute_variance_metrics: bool,
#     compute_featurewise_density_statistics: bool,
#     verbose: bool = False,
# ):
#     """
#     Partial code from sae_lens.evals.get_sparsity_and_variance_metrics
#     """
#     metric_dict = {}
#     feature_metric_dict = {}

#     if compute_l2_norms:
#         metric_dict["l2_norm_in"] = []
#         metric_dict["l2_norm_out"] = []
#         metric_dict["l2_ratio"] = []
#         metric_dict["relative_reconstruction_bias"] = []
#     if compute_sparsity_metrics:
#         metric_dict["l0"] = []
#         metric_dict["l1"] = []

#     mean_sum_of_squares = []  # for explained variance
#     mean_act_per_dimension = []  # for explained variance
#     mean_sum_of_resid_squared = []  # for explained variance
#     if compute_variance_metrics:
#         # explained_variance is left out of the dict here, since we don't want to naively
#         # average over the batch dimension. This is handled later in the function.
#         metric_dict["explained_variance_legacy"] = []
#         metric_dict["mse"] = []
#         metric_dict["cossim"] = []
#     if compute_featurewise_density_statistics:
#         feature_metric_dict["feature_density"] = []
#         feature_metric_dict["consistent_activation_heuristic"] = []

#     total_feature_acts = torch.zeros(model.sae.cfg.d_sae, device=model.sae.device)
#     total_feature_prompts = torch.zeros(model.sae.cfg.d_sae, device=model.sae.device)

#     batch_iter = range(n_batches)
#     if verbose:
#         batch_iter = tqdm(batch_iter, desc="Sparsity and Variance Batches")

#     for _ in batch_iter:
#         batch = next(iterable)

#         original_act = batch["embedding1"]  # shape (batch_size, d_in)

#         # normalise if necessary (necessary in training only, otherwise we should fold the scaling in)
#         original_act_scaled = model.activation_scaler.scale(original_act)

#         # send the (maybe normalised) activations into the SAE
#         sae_feature_activations = model.sae.encode(
#             original_act_scaled.to(device=model.sae.device)
#         )
#         sae_out_scaled = model.sae.decode(feature_act=sae_feature_activations).to(
#             device=model.sae.device
#         )
#         sae_out = model.activation_scaler.unscale(sae_out_scaled)

#         masked_sae_feature_activations = sae_feature_activations
#         flattened_sae_input = original_act
#         flattened_sae_feature_acts = sae_feature_activations
#         flattened_sae_out = sae_out

#         if compute_l2_norms:
#             l2_norm_in = torch.norm(flattened_sae_input, dim=-1)
#             l2_norm_out = torch.norm(flattened_sae_out, dim=-1)
#             l2_norm_in_for_div = l2_norm_in.clone()
#             l2_norm_in_for_div[l2_norm_in_for_div < 0.0001] = 1
#             l2_norm_ratio = l2_norm_out / l2_norm_in_for_div

#             # Equation 10 from https://arxiv.org/abs/2404.16014
#             # https://github.com/saprmarks/dictionary_learning/blob/main/evaluation.py
#             x_hat_norm_squared = torch.norm(flattened_sae_out, dim=-1) ** 2
#             x_dot_x_hat = (flattened_sae_input * flattened_sae_out).sum(dim=-1)
#             relative_reconstruction_bias = (
#                 x_hat_norm_squared.mean() / x_dot_x_hat.mean()
#             ).unsqueeze(0)

#             metric_dict["l2_norm_in"].append(l2_norm_in)
#             metric_dict["l2_norm_out"].append(l2_norm_out)
#             metric_dict["l2_ratio"].append(l2_norm_ratio)
#             metric_dict["relative_reconstruction_bias"].append(
#                 relative_reconstruction_bias
#             )

#         if compute_sparsity_metrics:
#             l0 = (flattened_sae_feature_acts > 0).sum(dim=-1).float()
#             l1 = flattened_sae_feature_acts.sum(dim=-1)
#             metric_dict["l0"].append(l0)
#             metric_dict["l1"].append(l1)

#         if compute_variance_metrics:
#             resid_sum_of_squares = (
#                 (flattened_sae_input - flattened_sae_out).pow(2).sum(dim=-1)
#             )

#             mse = resid_sum_of_squares
#             # Explained variance (old, incorrect, formula)
#             batched_variance_sum = (
#                 (flattened_sae_input - flattened_sae_input.mean(dim=0))
#                 .pow(2)
#                 .sum(dim=-1)
#             )
#             explained_variance_legacy = 1 - resid_sum_of_squares / batched_variance_sum
#             metric_dict["explained_variance_legacy"].append(explained_variance_legacy)
#             # Individual sums for the new (correct) formula. We're taking the mean over the batch
#             # dimension here to save memory, but we could also pass the full tensors and take the
#             # mean later (like we do for other metrics).
#             mean_sum_of_squares.append(
#                 (flattened_sae_input).pow(2).sum(dim=-1).mean(dim=0)  # scalar
#             )
#             mean_act_per_dimension.append(
#                 (flattened_sae_input).pow(2).mean(dim=0)  # [d_model]
#             )
#             mean_sum_of_resid_squared.append(resid_sum_of_squares.mean(dim=0))  # scalar
#             x_normed = flattened_sae_input / torch.norm(
#                 flattened_sae_input, dim=-1, keepdim=True
#             )
#             x_hat_normed = flattened_sae_out / torch.norm(
#                 flattened_sae_out, dim=-1, keepdim=True
#             )
#             cossim = (x_normed * x_hat_normed).sum(dim=-1)

#             metric_dict["mse"].append(mse)
#             metric_dict["cossim"].append(cossim)

#         if compute_featurewise_density_statistics:
#             sae_feature_activations_bool = (masked_sae_feature_activations > 0).float()
#             total_feature_acts += sae_feature_activations_bool.sum(dim=1).sum(dim=0)
#             total_feature_prompts += (sae_feature_activations_bool.sum(dim=1) > 0).sum(
#                 dim=0
#             )

#     # Aggregate scalar metrics
#     metrics: dict[str, float] = {}
#     for metric_name, metric_values in metric_dict.items():
#         metrics[f"{metric_name}"] = torch.cat(metric_values).mean().item()

#     # calculate explained variance
#     if compute_variance_metrics:
#         mean_sum_of_squares = torch.stack(mean_sum_of_squares).mean(dim=0)
#         mean_act_per_dimension = torch.cat(mean_act_per_dimension).mean(dim=0)
#         total_variance = mean_sum_of_squares - mean_act_per_dimension**2
#         residual_variance = torch.stack(mean_sum_of_resid_squared).mean(dim=0)
#         metrics["explained_variance"] = (1 - residual_variance / total_variance).item()

#     # Aggregate feature-wise metrics
#     feature_metrics: dict[str, list[float]] = {}
#     feature_metrics["consistent_activation_heuristic"] = (
#         total_feature_acts / total_feature_prompts
#     ).tolist()

#     return metrics, feature_metrics
