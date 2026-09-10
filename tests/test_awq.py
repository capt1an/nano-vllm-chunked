import unittest
from unittest.mock import patch
import torch
from nanovllm.layers.awq import AWQLinear, dequantize_awq, validate_awq


def pack(values):
    # Explicit exporter ordering, independent of the kernel's shift formula.
    values = values.reshape(values.shape[0], -1, 8)[..., [0, 2, 4, 6, 1, 3, 5, 7]].long()
    return (values << (torch.arange(8, device=values.device) * 4)).sum(-1).int()


class AWQTest(unittest.TestCase):
    def setUp(self):
        mock = patch("nanovllm.layers.awq.get_tensor_parallel_world_size", return_value=1)
        mock.start()
        self.addCleanup(mock.stop)

    def test_reference_and_loading(self):
        w = torch.arange(128 * 24).reshape(128, 24) % 16
        z = torch.arange(24).reshape(1, 24) % 16
        s = torch.full((1, 24), 0.125, dtype=torch.float16)
        torch.testing.assert_close(dequantize_awq(pack(w), pack(z), s), ((w-z)*0.125).half())
        layer = AWQLinear(128, [8, 16], ['q', 'k'])
        with self.assertRaisesRegex(ValueError, 'Missing'):
            layer.validate_loaded()
        for name, tensor in [('qweight', pack(w)), ('qzeros', pack(z)), ('scales', s)]:
            p = getattr(layer, name)
            split = 8 if name == 'scales' else 1
            p.weight_loader(p, tensor[:, :split], 'q')
            p.weight_loader(p, tensor[:, split:], 'k')
            torch.testing.assert_close(p, tensor)
        layer.validate_loaded()
        with self.assertRaisesRegex(ValueError, 'expected'):
            layer.qweight.weight_loader(layer.qweight, torch.zeros(128, 1), 'q')

    def test_reject_config(self):
        with self.assertRaises(ValueError):
            validate_awq(dict(quant_method='gptq'))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_kernel_and_graph(self):
        torch.manual_seed(7)
        with torch.device('cuda'):
            layer = AWQLinear(256, [72])
            w = torch.randint(0, 16, (256, 72))
            z = torch.randint(0, 16, (2, 72))
            s = (torch.rand(2, 72) * 0.05).half()
            for name, t in [('qweight', pack(w)), ('qzeros', pack(z)), ('scales', s)]:
                p = getattr(layer, name); p.weight_loader(p, t)
            ref = dequantize_awq(layer.qweight, layer.qzeros, layer.scales)
            for m in [1, 17, 64]:
                x = torch.randn(m, 256, dtype=torch.float16)
                expected = x @ ref
                torch.testing.assert_close(layer(x), expected, atol=0.008, rtol=0.008)
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                layer(x)
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(graph):
                y = layer(x)
            x.copy_(torch.randn_like(x))
            graph.replay()
            torch.testing.assert_close(y, x @ ref, atol=0.008, rtol=0.008)


if __name__ == '__main__':
    unittest.main()
