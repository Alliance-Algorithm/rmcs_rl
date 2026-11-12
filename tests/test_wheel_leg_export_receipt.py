import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "tool/export_wheel_leg_training_pair.py"
SPEC = importlib.util.spec_from_file_location("wheel_leg_export", MODULE_PATH)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class ExportReceiptTest(unittest.TestCase):
    def test_export_writes_paired_receipt_outside_model_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"test checkpoint")
            contract_path = root / "contract.json"
            contract_path.write_text("{}")
            target = root / "models/wheel_leg/policy.onnx"
            receipts = root / "model_receipts/wheel_leg"
            contract = {"model_spec_sha256": "model", "observation_contract_sha256": "obs",
                        "policy_hz": 50}
            actor = torch.nn.Linear(35, 6).eval()
            original_contract = exporter.CONTRACT
            exporter.CONTRACT = contract_path
            try:
                receipt = exporter.export(actor, actor, checkpoint, target, "wheelbipe",
                    updates=1, contract=contract, source_uri="test", receipt_output=receipts)
            finally:
                exporter.CONTRACT = original_contract
            receipt_path = receipts / "policy.deployment.json"
            self.assertTrue(target.is_file())
            self.assertFalse(target.with_suffix(".deployment.json").exists())
            self.assertEqual(json.loads(receipt_path.read_text()), receipt)
            self.assertEqual(receipt["onnx_sha256"], exporter.digest(target))
            self.assertFalse(receipt["behavior_qualified"])
            self.assertFalse(receipt["hardware_tested"])


if __name__ == "__main__":
    unittest.main()
