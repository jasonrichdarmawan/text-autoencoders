from typing import (
    Any,
)

from jaxtyping import Float

import torch

from torch.distributions.categorical import Categorical

import lightning as L

from sae_lens import (
    __version__,
)

from sae_lens.training.activation_scaler import ActivationScaler

from sae_lens.training.sae_trainer import (
    _update_sae_lens_training_version,
    _log_feature_sparsity,
    _unwrap_item,
)

from sae_lens.config import SAETrainerConfig


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
        cfg: SAETrainerConfig,
        sae: T_TRAINING_SAE,
    ):
        """
        Partial code from: sae_lens.training.sae_trainer.SAETrainer.__init__
        """
        super().__init__()

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

        self.save_hyperparameters(logger=False)

    def training_step(
        self,
        batch,
    ):
        """
        Partial code from sae_lens.training.sae_trainer.SAETrainer._train_step

        batch: dict with keys: see sae_utils.nllb_data_module.NLLBDataModule.train_dataloader
        """
        batch_size = (
            batch["nllb_200_6m_sample_embedding"]["embedding1"].shape[0]
            + batch["nllb_primary_datasets_embedding"]["embedding1"].shape[0]
        )
        self.n_training_samples += batch_size

        scaled_batch = self.activation_scaler(
            acts=torch.concat(
                [
                    batch["nllb_200_6m_sample_embedding"]["embedding1"],
                    batch["nllb_primary_datasets_embedding"]["embedding1"],
                ],
                dim=0,
            )
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
                self.log_dict({k: v for k, v in sparsity_log_dict.items()})
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
                self.log_dict(mean_train_step_log_dict)
            self.accumulated_train_step_log_dicts.clear()

        for scheduler in self.coefficient_schedulers.values():
            scheduler.step()

        # # Resample dead neurons
        # if (self.global_step + 1) % self.cfg.resample_freq == 0:
        #     self.resample_neurons(self.last_batch)

    def configure_optimizers(self):
        """
        Partial code from sae_lens.training.sae_trainer.SAETrainer.configure_optimizers
        """
        optimizer = torch.optim.AdamW(
            params=self.sae.parameters(),
            lr=self.cfg.lr,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
        )

        scheduler = {
            "scheduler": get_lr_scheduler(
                scheduler_name=self.cfg.lr_scheduler_name,
                lr=self.cfg.lr,
                optimizer=optimizer,
                warm_up_steps=self.cfg.lr_warm_up_steps,
                decay_steps=self.cfg.lr_decay_steps,
                training_steps=self.cfg.total_training_steps,
                lr_end=self.cfg.lr_end,
                num_cycles=self.cfg.n_restart_cycles,
            ),
            "interval": "step",
        }

        return [optimizer], [scheduler]

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
        return {
            "metrics/mean_log10_feature_sparsity": log_feature_sparsity.mean().item(),
            # on a log scale, how often features are active
            # see the [Interpreting the latent density histogram](https://arena-chapter1-transformer-interp.streamlit.app/[1.3.2]_Interpretability_with_SAEs#interpreting-the-latent-density-histogram)
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
