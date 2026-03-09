import gc
import time
import torch


def _stateless_init_process_group(master_address, master_port, rank, world_size, device):
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    return PyNcclCommunicator(pg, device=device)


class WorkerExtension:
    @staticmethod
    def _noise_seed(base_seed, param_idx, iid_noise):
        return int(base_seed) + int(param_idx) if iid_noise else int(base_seed)

    def perturb_self_weights(self, seed, noise_scale, negate=False, iid_noise=False):
        scale = float(noise_scale)
        sign = -1.0 if negate else 1.0
        for param_idx, (_, p) in enumerate(self.model_runner.model.named_parameters()):
            gen = torch.Generator(device=p.device)
            gen.manual_seed(self._noise_seed(seed, param_idx, iid_noise))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(sign * scale * noise)
            del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def restore_self_weights(self, seed, sigma, iid_noise=False):
        for param_idx, (_, p) in enumerate(self.model_runner.model.named_parameters()):
            gen = torch.Generator(device=p.device)
            gen.manual_seed(self._noise_seed(seed, param_idx, iid_noise))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(-float(sigma) * noise)
            del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def init_inter_engine_group(self, master_address: str, master_port: int, rank: int, world_size: int):
        self.inter_pg = _stateless_init_process_group(
            master_address, master_port, rank, world_size, self.device
        )
        return True

    def broadcast_all_weights(self, src_rank: int):
        for _, p in self.model_runner.model.named_parameters():
            self.inter_pg.broadcast(p, src=int(src_rank), stream=torch.cuda.current_stream())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def save_self_weights_to_disk(self, filepath):
        state_dict_to_save = {}
        for name, p in self.model_runner.model.named_parameters():
            state_dict_to_save[name] = p.detach().cpu()
        torch.save(state_dict_to_save, filepath)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(0.1)
        return True
