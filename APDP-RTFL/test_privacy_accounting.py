import unittest

import numpy as np

from privacy_accounting import RDPAccountant, calibrate_noise_multiplier
from fl_client import FLClient

try:
    import torch
    from baselines import PrivacyRuntimeConfig
    from torch_baselines import TorchLinearClient
except ImportError:
    torch = None
    PrivacyRuntimeConfig = None
    TorchLinearClient = None


class PrivacyAccountingTests(unittest.TestCase):
    def test_calibrated_schedule_stays_within_target(self):
        sigma = calibrate_noise_multiplier(0.1, 40, 5.0, 1e-5)
        accountant = RDPAccountant(5.0, 1e-5)
        before = accountant.epsilon
        event = accountant.spend(1, 20, 0.1, sigma)
        self.assertEqual(event.status, "spent")
        self.assertGreater(accountant.epsilon, before)
        self.assertLessEqual(accountant.epsilon, 5.0 + 1e-8)
        accountant.spend(2, 20, 0.1, sigma)
        self.assertLessEqual(accountant.epsilon, 5.0 + 1e-8)

    def test_overspend_is_rejected(self):
        accountant = RDPAccountant(1.0, 1e-5)
        event = accountant.spend(1, 100, 1.0, 0.1)
        self.assertEqual(event.status, "budget_exhausted")
        self.assertEqual(accountant.epsilon, 0.0)

    def test_empty_spend_is_safe(self):
        accountant = RDPAccountant(5.0, 1e-5)
        event = accountant.spend(1, 0, 0.2, 1.0)
        self.assertEqual(event.status, "spent")
        self.assertEqual(event.incremental_epsilon, 0.0)
        self.assertTrue(np.isfinite(accountant.epsilon))

    def test_multiclass_dp_sgd_update_shapes(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(12, 4))
        y = np.asarray([0, 1, 2] * 4)
        client = FLClient("test", X, y, 4, classes=np.asarray([0, 1, 2]), dp_batch_size=4, random_state=7)
        params = {"coef_": np.zeros((3, 4)), "intercept_": np.zeros(3)}
        client.set_global_model_parameters(params)
        accountant = RDPAccountant(5.0, 1e-5)
        delta, proof = client.train(
            epochs=1, global_params=params, privacy_accountant=accountant,
            noise_multiplier=20.0, round_num=1,
        )
        self.assertEqual(delta["coef_"].shape, (3, 4))
        self.assertEqual(delta["intercept_"].shape, (3,))
        self.assertIsNotNone(proof)
        self.assertGreater(accountant.epsilon, 0.0)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_torch_multiclass_dp_sgd_uses_accountant(self):
        rng = np.random.default_rng(11)
        X = rng.normal(size=(12, 4)).astype(np.float32)
        y = np.asarray([0, 1, 2] * 4)
        privacy_config = PrivacyRuntimeConfig(dp_batch_size=4, dp_l2_norm_clip=1.0)
        client = TorchLinearClient(
            "torch_test",
            X,
            y,
            4,
            classes=np.asarray([0, 1, 2]),
            device=torch.device("cpu"),
            batch_size=4,
            random_state=11,
            privacy_config=privacy_config,
        )
        params = {"coef_": np.zeros((3, 4), dtype=np.float32), "intercept_": np.zeros(3, dtype=np.float32)}
        accountant = RDPAccountant(5.0, 1e-5)
        delta, proof = client.train(
            global_params=params,
            epochs=1,
            use_dp=True,
            privacy_accountant=accountant,
            noise_multiplier=20.0,
            round_num=1,
        )
        self.assertEqual(delta["coef_"].shape, (3, 4))
        self.assertEqual(delta["intercept_"].shape, (3,))
        self.assertIsNotNone(proof)
        self.assertIsNotNone(client.last_privacy_event)
        self.assertEqual(client.last_privacy_event.status, "spent")
        self.assertGreater(accountant.epsilon, 0.0)


if __name__ == "__main__":
    unittest.main()
