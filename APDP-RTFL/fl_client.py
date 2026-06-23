import random
from sklearn.linear_model import SGDClassifier
import numpy as np
from dss import DifferentialStateSynchronizer
from zkip import ZeroKnowledgeIntegrityProofs
from ebcd import EntropyBasedCorruptionDetection
from earlystop import EarlyStopping
from sklearn.metrics import accuracy_score, log_loss
from privacy_accounting import RDPAccountant

class FLClient:
    def __init__(self, client_id, X_train, y_train, num_features, learning_rate=0.01,
                 dp_epsilon=1.0, dp_delta=1e-5, dp_l2_norm_clip=1.0, random_state=None,
                 X_val=None, y_val=None, earlystop_patience=3, classes=None, dp_batch_size=256):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.classes = np.asarray(classes if classes is not None else np.unique(y_train), dtype=int)
        if self.classes.size == 0:
            self.classes = np.array([0, 1])
        self.n_classes = len(self.classes)
        self.param_classes = 1 if self.n_classes <= 2 else self.n_classes
        self.model = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=learning_rate, random_state=random_state, warm_start=True)
        self.random_state = random_state
        self.num_features = num_features
        if self.X_train.shape[0] > 0 and self.X_train.shape[1] > 0 :
            self.model.coef_ = np.zeros((self.param_classes, num_features))
            self.model.intercept_ = np.zeros(self.param_classes)
            self.model.partial_fit(self.X_train[:1], self.y_train[:1], classes=self.classes)
        self.dss = DifferentialStateSynchronizer() 
        self.zkip = ZeroKnowledgeIntegrityProofs() 
        self.ebcd = EntropyBasedCorruptionDetection() 
        self.is_faulty = False
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.dp_l2_norm_clip = dp_l2_norm_clip
        self.dp_batch_size = int(dp_batch_size)
        self.X_val = X_val
        self.y_val = y_val
        self.earlystop_patience = earlystop_patience
        # 用于记录验证指标变化（供adaptive_adjustment使用）
        self.last_val_acc_before = None
        self.last_val_loss_before = None
        self.last_val_acc_gain = 0.0
        self.last_val_loss_drop = 0.0
        self.last_privacy_event = None

    def set_global_model_parameters(self, global_params):
        current_params = {}
        if global_params and 'coef_' in global_params and 'intercept_' in global_params:
            current_params = {
                'coef_': np.copy(global_params['coef_']).reshape(self.param_classes, self.num_features),
                'intercept_': np.copy(global_params['intercept_'])
            }
        else:
            current_params = {
                'coef_': np.zeros((self.param_classes, self.num_features)),
                'intercept_': np.zeros(self.param_classes)
            }
        current_params['coef_'] = np.copy(current_params['coef_']).reshape(self.param_classes, self.num_features)
        current_params['intercept_'] = np.copy(current_params['intercept_']).reshape(self.param_classes)
        self.model.coef_ = current_params['coef_']
        self.model.intercept_ = current_params['intercept_']
        self.dss.set_base_model_parameters(current_params)

    def _apply_differential_privacy(self, delta_params):
        noisy_delta_params = {}
        total_norm = 0.0
        for key in delta_params:
            total_norm += np.linalg.norm(delta_params[key].flatten())**2
        total_norm = np.sqrt(total_norm)
        clip_factor = min(1.0, self.dp_l2_norm_clip / (total_norm + 1e-6))
        noise_stddev = (self.dp_l2_norm_clip * np.sqrt(2 * np.log(1.25 / self.dp_delta))) / self.dp_epsilon
        if self.dp_epsilon == 0:
             noise_stddev = 0
        for key in delta_params:
            clipped_delta = delta_params[key] * clip_factor
            noise = np.random.normal(0, noise_stddev, size=delta_params[key].shape)
            noisy_delta_params[key] = clipped_delta + noise
        return noisy_delta_params

    def _train_with_dp_sgd(self, epochs, global_params, fedprox_mu, accountant, noise_multiplier, round_num):
        """Run client-local, sample-level DP-SGD for the multiclass linear head."""
        if self.n_classes <= 2:
            raise ValueError("DP-SGD currently requires a multiclass linear head")
        n_samples = len(self.y_train)
        batch_size = min(max(1, self.dp_batch_size), n_samples)
        sample_rate = batch_size / n_samples
        steps = int(epochs) * int(np.ceil(n_samples / batch_size))
        if accountant is not None:
            event = accountant.spend(round_num or 0, steps, sample_rate, noise_multiplier)
            self.last_privacy_event = event
            if event.status != "spent":
                return None, None

        weights = np.copy(global_params['coef_'])
        bias = np.copy(global_params['intercept_'])
        class_to_index = {int(label): idx for idx, label in enumerate(self.classes)}
        targets = np.asarray([class_to_index[int(label)] for label in self.y_train], dtype=int)
        rng = np.random.default_rng((self.random_state or 0) + 10007 * int(round_num or 1))
        noise_stddev = noise_multiplier * self.dp_l2_norm_clip
        for _ in range(int(epochs)):
            for _ in range(int(np.ceil(n_samples / batch_size))):
                selected = rng.random(n_samples) < sample_rate
                if not np.any(selected):
                    selected[rng.integers(0, n_samples)] = True
                X_batch, y_batch = self.X_train[selected], targets[selected]
                logits = X_batch @ weights.T + bias
                logits -= np.max(logits, axis=1, keepdims=True)
                probabilities = np.exp(logits)
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                residual = probabilities
                residual[np.arange(len(y_batch)), y_batch] -= 1.0
                grad_w_each = residual[:, :, None] * X_batch[:, None, :]
                grad_b_each = residual
                norms = np.sqrt(np.sum(grad_w_each**2, axis=(1, 2)) + np.sum(grad_b_each**2, axis=1))
                scales = np.minimum(1.0, self.dp_l2_norm_clip / (norms + 1e-12))
                grad_w = np.mean(grad_w_each * scales[:, None, None], axis=0)
                grad_b = np.mean(grad_b_each * scales[:, None], axis=0)
                denominator = max(1, len(y_batch))
                grad_w += rng.normal(0.0, noise_stddev / denominator, size=grad_w.shape)
                grad_b += rng.normal(0.0, noise_stddev / denominator, size=grad_b.shape)
                if fedprox_mu > 0:
                    grad_w += fedprox_mu * (weights - global_params['coef_'])
                    grad_b += fedprox_mu * (bias - global_params['intercept_'])
                weights -= self.model.eta0 * grad_w
                bias -= self.model.eta0 * grad_b
        self.model.coef_ = weights
        self.model.intercept_ = bias
        self.dss.set_base_model_parameters(global_params)
        delta_params = self.dss.compute_delta({'coef_': weights, 'intercept_': bias})
        proof = self.zkip.generate_proof(delta_params)
        return delta_params, proof

    def train(self, epochs, use_dp=True, fedprox_mu=0.0, global_params=None,
              privacy_accountant: RDPAccountant | None = None, noise_multiplier: float | None = None,
              round_num: int | None = None):
        #记录before
        if self.X_val is not None and self.y_val is not None and len(np.unique(self.y_val)) > 0 :
            # 记录训练前的验证指标
            acc_before, loss_before = self.evaluate_local_validation()
            self.last_val_acc_before = acc_before
            self.last_val_loss_before = loss_before
        else:
            self.last_val_acc_before = 0.0
            self.last_val_loss_before = 0.0
        if self.is_faulty or self.X_train.shape[0] == 0:
            return None, None 
        if not hasattr(self.model, 'classes_') and len(self.y_train) > 0:
            self.model.partial_fit(self.X_train[:1], self.y_train[:1], classes=self.classes)
        if use_dp and noise_multiplier is not None:
            return self._train_with_dp_sgd(epochs, global_params, fedprox_mu, privacy_accountant, noise_multiplier, round_num)
        earlystop = EarlyStopping(mode='min', patience=self.earlystop_patience)
        for epoch in range(epochs):
            self.model.partial_fit(self.X_train, self.y_train)
            if fedprox_mu > 0 and global_params is not None:
                self.model.coef_ -= fedprox_mu * (self.model.coef_ - global_params['coef_'])
                self.model.intercept_ -= fedprox_mu * (self.model.intercept_ - global_params['intercept_'])
            # Early stopping: check validation loss if validation data is provided
            if self.X_val is not None and self.y_val is not None and len(np.unique(self.y_val)) >= 2:
                try:
                    val_pred = self.model.predict_proba(self.X_val)
                    val_loss = log_loss(self.y_val, val_pred, labels=self.classes)
                except Exception:
                    val_loss = float('inf')
                if earlystop.step(val_loss, {'coef_': np.copy(self.model.coef_), 'intercept_': np.copy(self.model.intercept_)}):
                    break
        # Restore best params if early stopped
        best_params = earlystop.get_best()
        if best_params is not None:
            self.model.coef_ = best_params['coef_']
            self.model.intercept_ = best_params['intercept_']
        current_params = {'coef_': self.model.coef_, 'intercept_': self.model.intercept_}
        try:
            delta_params = self.dss.compute_delta(current_params)
        except ValueError as e:
            self.dss.set_base_model_parameters({'coef_': np.zeros_like(current_params['coef_']), 
                                                'intercept_': np.zeros_like(current_params['intercept_'])})
            delta_params = self.dss.compute_delta(current_params)
        if use_dp and self.dp_epsilon > 0:
            noisy_delta_params = self._apply_differential_privacy(delta_params)
        else:
            noisy_delta_params = delta_params
        # 训练结束后再次评估，计算提升量
        if self.X_val is not None and self.y_val is not None and len(self.y_val) > 0:
            acc_after, loss_after = self.evaluate_local_validation()
            self.last_val_acc_gain = acc_after - self.last_val_acc_before if self.last_val_acc_before is not None else 0.0
            self.last_val_loss_drop = self.last_val_loss_before - loss_after if self.last_val_loss_before is not None else 0.0
        else:
            self.last_val_acc_gain = 0.0
            self.last_val_loss_drop = 0.0
        proof = self.zkip.generate_proof(noisy_delta_params)
        return noisy_delta_params, proof

    def evaluate_local_validation(self):
        """
        在本地验证集上评估当前模型，返回 (accuracy, binary_cross_entropy_loss)
        """
        if self.X_val is None or self.y_val is None or len(self.y_val) == 0:
            return 0.0, 0.0
        try:
            # 预测
            predictions = self.model.predict(self.X_val)
            acc = accuracy_score(self.y_val, predictions)
            # 计算 binary cross-entropy loss
            if hasattr(self.model, "predict_proba"):
                probas = np.clip(self.model.predict_proba(self.X_val), 1e-15, 1 - 1e-15)
                row_sums = probas.sum(axis=1, keepdims=True)
                probas = probas / np.where(row_sums == 0, 1.0, row_sums)
                loss = log_loss(self.y_val, probas, labels=self.classes)
            else:
                loss = 0.0  # 无法计算 loss 时用 0 占位
            return acc, loss
        except Exception as e:
            print(f"Client {self.client_id} validation error: {e}")
            return 0.0, 0.0

    def simulate_failure(self, probability=0.1):
        if random.random() < probability:
            self.is_faulty = True
        else:
            self.is_faulty = False
        return self.is_faulty

    def model_parameters(self):
        return {'coef_': self.model.coef_, 'intercept_': self.model.intercept_}
