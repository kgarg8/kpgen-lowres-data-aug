import json, torch as T, pdb
from tqdm import tqdm
from utils.data_utils import count_iterations

class Trainer:
    def __init__(self, agent, sample_len, args, config, logpaths=None, stats_path=None, global_step=None, desc=None, no_display=False, display_fn=None, example_display_fn=None):

        self.config               = config
        self.args                 = args
        self.global_step          = global_step
        self.agent                = agent
        self.sample_len           = sample_len
        self.generator_len        = count_iterations(sample_len, config["batch_size"])
        self.desc                 = desc
        self.no_display           = no_display
        self.display_fn           = display_fn
        self.display_step         = args.display_step
        self.example_display_fn   = example_display_fn
        self.example_display_step = args.example_display_step
        self.logpaths             = logpaths
        self.stats_path           = stats_path
        self.total_batch_size     = config["batch_size"]

    def model_state_dict(self):
        return self.agent.model.state_dict()

    def optimizer_state_dict(self):
        return self.agent.optimizer.state_dict()

    def run(self, epoch, DataLoader, train, current_iter=None):

        tracked_items = []
        i = 0
        with tqdm(total=self.generator_len, desc=self.desc, position=0, disable=self.no_display) as pbar:

            cumsum_batch_size = 0

            for batches in DataLoader:
                for batch_id, batch in enumerate(batches):
                    item = self.agent.run(batch, train=train)
                    if train:
                        loss      = item["loss"]
                        accu_step = self.total_batch_size // batch["batch_size"]
                        self.agent.backward(loss / accu_step)
                    cumsum_batch_size += batch["batch_size"]
                    item = {k: v for k, v in item.items() if k != "loss"}
                    tracked_items.append(item)

                    if (cumsum_batch_size // self.total_batch_size == i + 1) or (cumsum_batch_size == self.sample_len) or batch_id == len(batches) - 1:

                        if train:
                            self.agent.step()
                            self.global_step += 1
                            #print(self.agent.optimizer.param_groups[0]["lr"])
                            #print(self.global_step, ' ', i, ' ', self.total_batch_size)

                        if current_iter is not None:
                            iter = i + current_iter 
                        else:
                            iter = i
                        if self.display_fn is not None:
                            display_string = self.display_fn(epoch=epoch, iter=iter, item=item, args=self.args, config=self.config)
                            if self.logpaths is not None:
                                with open(self.logpaths["verbose_log_path"], "a") as fp:
                                    fp.write(display_string + "\n")

                        if self.display_step is not None and self.display_fn is not None and not self.no_display:
                            if i % self.display_step == 0:
                                pbar.write('lr:' + str(self.agent.optimizer.param_groups[0]["lr"]) + ' ' + display_string)
                                if self.logpaths is not None:
                                    with open(self.logpaths["log_path"], "a") as fp:
                                        fp.write('lr:' + str(self.agent.optimizer.param_groups[0]["lr"]) + ' ' + display_string + "\n")

                        if self.example_display_fn is not None:
                            display_string = self.example_display_fn(epoch=epoch, iter=iter, item=item, args=self.args, config=self.config)
                            if self.logpaths is not None:
                                with open(self.logpaths["verbose_log_path"], "a") as fp:
                                    fp.write(display_string + "\n")

                        if self.example_display_step is not None and self.example_display_fn is not None and not self.no_display:
                            if i % self.example_display_step == 0:
                                pbar.write(display_string)
                                if self.logpaths is not None:
                                    with open(self.logpaths["log_path"], "a") as fp:
                                        fp.write(display_string + "\n")

                        if self.stats_path is not None:
                            with open(self.stats_path, "a") as fp:
                                stats_string = "\n".join([json.dumps(x) for x in item["stats_metrics"]])
                                fp.write(stats_string + "\n")

                        i += 1
                        if not self.no_display:
                            pbar.update(1)

        return tracked_items

    def regenerate_generator_len(self):
        self.generator_len = count_iterations(self.sample_len, self.config["batch_size"])

    def train(self, epoch, DataLoader, current_iter=None):
        self.total_batch_size = self.config["batch_size"]
        self.generator_len    = count_iterations(self.sample_len, self.config["batch_size"])
        tracked_items         = self.run(epoch, DataLoader, train=True, current_iter=current_iter)
        return tracked_items

    def eval(self, epoch, DataLoader):
        self.total_batch_size = self.config["dev_batch_size"]
        self.generator_len    = count_iterations(self.sample_len, self.config["dev_batch_size"])
        with T.no_grad():
            tracked_items = self.run(epoch, DataLoader, train=False)
        return tracked_items