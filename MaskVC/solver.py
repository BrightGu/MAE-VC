import os
import time
import torch
from torch.backends import cudnn
import numpy as np
import yaml
from torch.utils.data import DataLoader
from MaskVC.data.ge2e_dataset import get_data_loaders
from MaskVC import util
from MaskVC.meldataset_e2e2 import Test_MelDataset, get_test_dataset_filelist,mel_denormalize
from hifivoice.inference_e2e import  hifi_infer
from MaskVC.modules.Base.base_6c_16b_36_FSC_50 import MagicModel
# used for print log
label_watch = "Base.base_6c_16b_36_FSC_50"

class Solver():
	def __init__(self, config):
		super(Solver, self).__init__()
		self.config = config
		self.local_rank = self.config['local_rank']
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.batch_size = self.config['batch_size']
		self.utters_num = self.config['utters_num']
		self.make_records()
		# train
		self.total_steps = self.config['total_steps']
		self.save_steps = self.config['save_steps']
		self.eval_steps = self.config['eval_steps']
		self.log_steps = self.config['log_steps']
		self.schedule_steps = self.config["schedule_steps"]
		self.warmup_steps = self.config["warmup_steps"]

		self.learning_rate = self.config["learning_rate"]
		self.learning_rate_min = self.config["learning_rate_min"]
		self.Generator = MagicModel().to(self.device)
		self.train_data_loader = get_data_loaders(self.config['label_clip_mel_pkl'],self.batch_size,self.utters_num)

		self.optimizer = torch.optim.AdamW(
			[{'params': self.Generator.parameters(), 'initial_lr': self.config["learning_rate"]}],
			self.config["learning_rate"],betas=[self.config["adam_b1"], self.config["adam_b2"]])
		self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.schedule_steps,
																	eta_min=self.learning_rate_min)
		self.criterion = torch.nn.L1Loss()
		self.init_epoch = 0
		if self.config['resume']:
			self.resume_model(self.config['resume_num'])
		self.logging.info('config = %s', self.config)
		print('param Generator size = %fM ' % (util.count_parameters_in_M(self.Generator)))

	def make_records(self):
		time_record = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
		self.log_dir = os.path.join(self.config['out_dir'],time_record,"log")
		self.model_dir = os.path.join(self.config['out_dir'],time_record,"model")
		self.write_dir = os.path.join(self.config['out_dir'],time_record,"write")
		self.convt_mel_dir = os.path.join(self.config['out_dir'],time_record,"infer","mel")
		self.convt_voice_dir = os.path.join(self.config['out_dir'],time_record,"infer","voice")
		os.makedirs(self.log_dir, exist_ok=True)
		os.makedirs(self.model_dir, exist_ok=True)
		os.makedirs(self.write_dir, exist_ok=True)
		os.makedirs(self.convt_mel_dir, exist_ok=True)
		os.makedirs(self.convt_voice_dir, exist_ok=True)
		self.logging = util.Logger(self.log_dir, "log.txt")


	def get_test_data_loaders(self):
		test_filelist = get_test_dataset_filelist(self.config["test_wav_dir"])
		testset = Test_MelDataset(test_filelist,self.config["n_fft"],self.config["num_mels"],
							 self.config["hop_size"], self.config["win_size"], self.config["sampling_rate"],self.config["fmin"],
							 self.config["fmax"], device=self.device)
		test_data_loader = DataLoader(testset, num_workers=1, shuffle=False, sampler=None,
									  batch_size=1, pin_memory=False, drop_last=True)
		return test_data_loader


	def resume_model(self, resume_num):
		print("*********  [load]   ***********")
		checkpoint_file = os.path.join(self.model_dir, 'checkpoint-%d.pt' % (resume_num))
		self.logging.info('loading the model from %s' % (checkpoint_file))
		checkpoint = torch.load(checkpoint_file, map_location='cpu')
		# start epoch and
		self.init_epoch = checkpoint['epoch']
		self.Generator.load_state_dict(checkpoint['Generator'])
		self.optimizer.load_state_dict(checkpoint['optimizer'])
		self.scheduler.load_state_dict(checkpoint['scheduler'])

	def reset_grad(self):
		self.optimizer.zero_grad()

	def warmup_lr(self, total_step):
		lr = self.learning_rate_min + (total_step/self.warmup_steps)*self.learning_rate
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr


	def train(self):
		self.Generator.train()
		self.logging.info('【%s】',label_watch)
		for step in range(self.total_steps):
			batch = next(self.train_data_loader)
			feats = batch[0].to(self.device)
			spk_feats = batch[1].to(self.device)
			out = self.Generator.infer(feats, spk_feats)
			loss = self.criterion(out,spk_feats)
			self.reset_grad()
			loss.backward()
			self.optimizer.step()
			if step % self.log_steps == 0:
				lr = self.optimizer.state_dict()['param_groups'][0]['lr']
				self.logging.info('【train %d】lr:  %.10f', step, lr)
				self.logging.info('【train_%d】loss:%f',step, loss)
			if step < self.warmup_steps:
				self.warmup_lr(step)
			else:
				self.scheduler.step()
			if (step+1) % self.save_steps == 0 or step == self.total_steps-1:
				save_model_path = os.path.join(self.model_dir, 'checkpoint-%d.pt' % (step))
				self.logging.info('saving the model to the path:%s', save_model_path)
				torch.save({'step': step+1,
							'config': self.config,
							'Generator': self.Generator.state_dict(),
							'optimizer': self.optimizer.state_dict(),
							'scheduler': self.scheduler.state_dict()
							},
						   save_model_path, _use_new_zipfile_serialization=False)
				self.infer(step)

	# test during training
	def infer(self,epoch):
		test_data_loader = self.get_test_data_loaders()
		self.Generator.eval()
		mel_npy_file_list=[]
		with torch.no_grad():
			for idx, (src_mel, tar_mel, convert_label) in enumerate(test_data_loader):
				src_mel = src_mel.cuda()
				tar_mel = tar_mel.cuda()
				fake_mel = self.Generator.infer(src_mel,tar_mel)
				fake_mel = torch.clamp(fake_mel, min=0, max=1)
				fake_mel = mel_denormalize(fake_mel)
				fake_mel = fake_mel.transpose(1,2)
				fake_mel = fake_mel.detach().cpu().numpy()
				file_name = "epoch"+str(epoch)+"_"+convert_label[0]
				mel_npy_file = os.path.join(self.convt_mel_dir, file_name+ '.npy')
				# mel_npy_list.append(mel_npy_file)
				np.save(mel_npy_file, fake_mel, allow_pickle=False)
				mel_npy_file_list.append([file_name,fake_mel])
		hifi_infer(mel_npy_file_list,self.convt_voice_dir)
		self.Generator.train()

def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
	print("【Solver  %s】"%label_watch)
	cudnn.benchmark = True
	config_path = r"MaskVC/hifi_config.yaml"
	with open(config_path) as f:
		config = yaml.load(f, Loader=yaml.Loader)
	solver = Solver(config)
	solver.train()


