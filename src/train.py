import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

from .model import MLP       
from .dataset import generate_gaussian_mixture
from .schedule import betas, alphas, alpha_bars, ts_desc
from .config import DEVICE, TRAINING, DATASETS, N_DIFFUSION_STEPS, CKPT_DIR

device = DEVICE
n_steps = TRAINING["n_steps"]
batch_size = TRAINING["batch_size"]
lr = TRAINING["lr"]
ckpt_dir = CKPT_DIR
n_diffusion_steps = N_DIFFUSION_STEPS

torch.manual_seed(42)     

def train_model(name, dataset_config, existing_checkpoint=None, load_file=None, k=1.0, log_every=5000, is_ebm = False):
	"""Trains model according to a dataset defined by dataset_config"""

	x0_all = generate_gaussian_mixture(
        n_samples=dataset_config["dataset_shape"][0],
		means=dataset_config["means"],
		stds=dataset_config["stds"],
        device='cpu',
    )

	loader = DataLoader(
		TensorDataset(x0_all),
		batch_size=batch_size,
		shuffle=True,
		drop_last=True,
		num_workers=0,      # keep simple; set >0 if you want
		pin_memory=True,    # helps H2D transfer
	)

	model = MLP().to(device)
	if existing_checkpoint is not None:
		print(f"Loading {existing_checkpoint}")
		checkpoint = torch.load(existing_checkpoint, map_location=device)
		model.load_state_dict(checkpoint)
		model.eval()

	opt = optim.Adam(model.parameters(), lr=lr)
	model.train()

	data_iter = iter(loader)

	for step in range(1, n_steps + 1):
		try:
			(x0,) = next(data_iter)
		except StopIteration:
			data_iter = iter(loader)  # reshuffles because shuffle=True
			(x0,) = next(data_iter)

		x0 = x0.to(device, non_blocking=True)

		t = torch.randint(0, n_diffusion_steps, (batch_size, 1), device=device)
		a_bar = alpha_bars[t]
		
		noise = torch.randn_like(x0)
		xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise

		xt = xt.detach().requires_grad_(True)  # we need grads wrt xt
		output = model(xt, t)   # energy values from EBM: shape [B, 1] or [B]

		if is_ebm:  

			grad_xt = torch.autograd.grad(
				outputs=output,
				inputs=xt,
				grad_outputs=torch.ones_like(output),
				create_graph=True,       # keep graph if you need higher-order grads
				retain_graph=True        # if you will use output again
			)[0]

			eps_hat = -grad_xt
		else:
			eps_hat = output 

		loss = ((noise - eps_hat) ** 2).mean()

		opt.zero_grad()
		loss.backward()
		opt.step()

		if step % log_every == 0:
			print(f"temperature={k} step={step} loss={loss.item():.4f}")

	save_path = f"{ckpt_dir}/{name}_{k:.1f}_{is_ebm}.pt"
	torch.save(model.state_dict(), save_path)
	print(f"Trained model saved to {save_path}")

	return model


if __name__ == "__main__":	

	dataset_names = ["composed"]
	k = 1.0
	is_ebm = True

	for name in dataset_names:
		
		dataset_config = DATASETS[name]
		existing_checkpoint = f"{ckpt_dir}/{name}_{k:.1f}_{is_ebm}.pt"
		
		if os.path.exists(existing_checkpoint):
			model = train_model(name, dataset_config, existing_checkpoint, is_ebm = is_ebm)
		else:
			model = train_model(name, dataset_config, is_ebm = is_ebm)